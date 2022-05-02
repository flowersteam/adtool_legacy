from time import sleep
from AutoDiscServer.experiments import BaseExperiment
from AutoDiscServer.utils import ExperimentStatusEnum, list_profiles, parse_profile, match_except_number
from AutoDiscServer.utils.DB import AppDBLoggerHandler, AppDBMethods, AppDBCaller
from AutoDiscServer.utils.DB.expe_db_utils import serialize_autodisc_space
import threading
import os
from pexpect import pxssh
import json
import pickle
import traceback
from copy import copy


class RemoteExperiment(BaseExperiment):
    '''
        Remote experiment that packages the auto_disc lib along with configuration files and sends it to remote server.
        Pipelines are run locally on remote server (everything is stored on disk). This class monitors progress and downloads produced files.
    '''

    def __init__(self, host_profile_name, *args, **kwargs):
        self.__host_profile = parse_profile(next(profile[1] for profile in list_profiles() if profile[0] == host_profile_name))

        args[1]["logger_handlers"] = [{"name": "logFile", "config": {"folder_log_path": self.__host_profile["work_path"]+"/logs/"}}]
        args[1]["callbacks"] = {"on_discovery": [
                                {"name": "disk", 
                                "config": {"to_save_outputs": args[1]["experiment"]["config"]["discovery_saving_keys"], 
                                           "folder_path": self.__host_profile["work_path"]+"/outputs/"
                                          }
                                }
                              ], 
            "on_save_finished": [],
            "on_cancelled": [], 
            "on_finished": [], 
            "on_error": [], 
            "on_saved": [{"name": "disk", "config": {"folder_path": self.__host_profile["work_path"]+"/checkpoints/"}}]
            }
        super().__init__(*args, **kwargs)


        self.nb_seeds_finished = 0
        
        self.app_db_logger_handler = AppDBLoggerHandler('http://127.0.0.1:3000', self.id, self._get_current_checkpoint_id)
        
        self.port = 22

        self.threading_lock = threading.Lock()

        ### create connection
        self.shell = pxssh.pxssh()
        self.ssh_config_file_path = "/home/mperie/.ssh/config" # TODO change to a correct file path
        self.shell.login(self.__host_profile["ssh_configuration"], ssh_config=self.ssh_config_file_path)

        self._app_db_caller = AppDBCaller("http://127.0.0.1:3000")

    def __close_ssh(self):
        self._monitor_async.join()
        self.shell.close()
        if self.killer_shell is not None:
            self.killer_shell.close()


#region public launching
    def prepare(self):
        # push to server
        self._send_packaged_experiment_to_remote()

    def start(self):
        # move to work directory
        self.shell.sendline('cd {}'.format(self.__host_profile["work_path"]))
        # make command we will execute to launch the experiment
        exec_command = self.__host_profile["execution"].replace("$NB_SEEDS", "{}".format(self.experiment_config["experiment"]["config"]["nb_seeds"]))
        exec_command = exec_command.replace(
            "$ARGS",
             "--config_file {}/parameters_remote.json --experiment_id {} --nb_iterations {}"
             .format(
                 self.__host_profile["work_path"], 
                 self.id, 
                 self.experiment_config["experiment"]["config"]["nb_iterations"]
             )
        )
        exec_command = exec_command.replace("$EXPE_ID", self.__host_profile["work_path"]+"/run_ids/"+str(self.id))

        # execute command
        self.shell.sendline(exec_command)
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "the command to run the experiment on the remote server has been launched"
                                }
                            )
        # get run id of each python who have been launched
        self.__run_id = self.__get_run_id() #TODO save run_id in db
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "the command has been taken into account by the server"
                                }
                            )
        # read log file to manege remote experiment
        self._monitor_async = threading.Thread(target=self._monitor)
        self._monitor_async.start()

    def reload(self):
        response = self._app_db_caller("/experiments?id=eq.{}".format(self.id), 
                                AppDBMethods.GET, {})
        self.__run_id =json.loads(response.content.decode())[0]['remote_run_id']
        self._monitor_async = threading.Thread(target=self._monitor)
        self._monitor_async.start()

    def stop(self):
        ### create new shell who will kills process launched by first shell
        ## open ssh connection
        self.killer_shell = pxssh.pxssh()
        self.killer_shell.login(self.__host_profile["ssh_configuration"], ssh_config=self.ssh_config_file_path)
        ## create and exec cancellation command
        cancellation_command = self.__host_profile["cancellation"].replace("$RUN_ID", self.__run_id)
        self.killer_shell.sendline(cancellation_command)
        ## declare all seed finished(all cancelled)
        self.threading_lock.acquire()
        self.nb_seeds_finished = self.experiment_config["experiment"]["config"]["nb_seeds"]
        self.threading_lock.release()
        ## close properly ssh connection
        self.__close_ssh()
        #update app db 
        self.callback_to_all_running_seeds(lambda seed, current_checkpoint_id : self.on_cancelled(seed))

#endregion

#region communicate files with remote

    def _send_packaged_experiment_to_remote(self):
        to_push_folder_path = self.__host_profile["local_tmp_path"]+"/remote_experiment/push_to_server" # where we saved remote files need to be pushed on server
        if(not os.path.exists(to_push_folder_path)):
            os.makedirs(to_push_folder_path)

        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "send necessary files to the remote server"
                                }
                            )

        ## push libs
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "package and send the library to the remote server"
                                }
                            )
        # make path
        lib_path =os.path.dirname(os.path.realpath(__file__))+"/../../../libs" 
        lib_path_tar = to_push_folder_path+"/libs.tar.gz"
        # push and untar lib
        self.__tar_local_folder(lib_path, lib_path_tar)
        self.__push_folder(lib_path_tar, self.__host_profile["work_path"])
        self.__untar_folder(self.__host_profile["work_path"]+"/libs.tar.gz", self.__host_profile["work_path"])

        ## push slurm file
        # make path
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "send config files to the remote server"
                                }
                            )
        additional_file_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../configs/remote_experiments/additional_files"
        additional_file_path_tar = to_push_folder_path+"/additional_files.tar.gz"
        self.__tar_local_folder(additional_file_path, additional_file_path_tar)
        self.__push_folder(additional_file_path_tar, self.__host_profile["work_path"])
        self.__untar_folder(self.__host_profile["work_path"]+"/additional_files.tar.gz", self.__host_profile["work_path"])

        ## push parameters file (json)
        # save json on disk
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "send parameters files to the remote server"
                                }
                            )
        json_file_path = to_push_folder_path + "/parameters_remote.json"
        with open(json_file_path, 'w+') as fp:
            json.dump(self.cleared_config, fp)
        # push json on remote server
        self.__push_folder(json_file_path, self.__host_profile["work_path"])
        
        # make logs folder on remote server
        self.shell.sendline("mkdir {}/{}".format(self.__host_profile["work_path"], "logs"))
        self.shell.sendline("mkdir {}/{}".format(self.__host_profile["work_path"], "run_ids"))

        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "All necessary files have been sent to the remote server and are ready to be used"
                                }
                            )

    def __tar_local_folder(self, folder_src, tar_path):
        folder_src_split = folder_src.split("/")
        target_folder = folder_src_split[-1] # get last child folder name
        # remove last child folder from path
        index = folder_src.rfind(target_folder)
        parent_folder = folder_src[:index]
        os.system('cd {} && tar -czvf {} {}'.format(parent_folder, tar_path, target_folder))

    def __push_folder(self, folder_path_local, folder_path_remote):
        os.system('scp -r {} {}:{}'.format(folder_path_local, self.__host_profile["ssh_configuration"], folder_path_remote))
    
    def __pull_folder(self, folder_path_remote, folder_path_local):
        os.system('scp -r {}:{} {}'.format(self.__host_profile["ssh_configuration"], folder_path_remote, folder_path_local))

    def __untar_folder(self, tar_path, folder_path_dest):
        self.shell.sendline('tar -xvf {} -C {} \n'.format(tar_path, folder_path_dest))

    def __pull_files(self, remote_path, sub_folders, run_idx, local_folder):
        """
        brief: pull all file create by a callback
        param: log : list
               remote_path: string : path
               sub_folders: string list : folder's name
               run_idx: string : run_idx
        """
        auto_disc_parent_folder = remote_path.find(remote_path.split('/')[-4])
        local_folder += remote_path[auto_disc_parent_folder-1:]
        for sub_folder in sub_folders:
            if not os.path.exists("{}{}/".format(local_folder, sub_folder)):
                os.makedirs("{}{}/".format(local_folder, sub_folder))
            self.__pull_folder("{}{}/idx_{}.*".format(remote_path, sub_folder, run_idx), "{}{}/".format(local_folder, sub_folder))
#endregion

#region monitor
    def _monitor(self):
        ### monitor remote experiment
        ## make path
        local_folder = self.__host_profile["local_tmp_path"]+"/remote_experiment/out" # where we saved remote experiments output before puting them in our db
        if(not os.path.exists(local_folder)):
            os.makedirs(local_folder)
        ## listen log file and do the appropriate action
        while not self.test_file_exist(self.__host_profile["work_path"]+"/logs/exp_{}.log".format(self.id)):
            self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "waiting for the server"
                                }
                            )
            sleep(self.__host_profile["check_experiment_launched_every"])
        self.shell.sendline(
            'tail -F -n +1 {}'
            .format(self.__host_profile["work_path"]+"/logs/exp_{}.log".format(self.id))
        )
        #change experiment status from preparing to running in db 
        response = self._app_db_caller("/experiments?id=eq.{}".format(self.id), 
                                AppDBMethods.PATCH, 
                                {"exp_status": ExperimentStatusEnum.RUNNING})
        self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "the experiment start"
                                }
                            )
        self.__listen_log_file(local_folder)

    def __parse_log(self, log, local_folder):
        """
        Brief: read current log line and choose the appropriate action
        param: log : string
        local_folder : string
        """
        is_current_seed_finished = False
        logger_name, log_level_name, seed_number, log_id, message = self.__get_log_part(log)
        if not self.__log_not_in_db(log_id):
            return logger_name, log_level_name, seed_number, log_id, message

        if self.__new_files_saved_on_remote_disk(message):
            remote_path, sub_folders, run_idx = self.__get_saved_files_to_pull(message)
            super().on_progress(seed_number, run_idx + 1)
            if sub_folders is not None:
                self.__pull_files(remote_path, sub_folders, run_idx, local_folder)
            if remote_path.split("/")[-4] == "outputs":
                self.save_discovery_to_expe_db(sub_folders=sub_folders, run_idx=run_idx, folder=local_folder, seed=seed_number, experiment_id=self.id)
            elif remote_path.split("/")[-4] == "checkpoints":
                self.save_modules_to_expe_db(sub_folders=sub_folders, run_idx=run_idx, folder=local_folder, seed=seed_number, experiment_id=self.id)
        elif self.__is_finished(message):
            super().on_finished(seed_number)
            is_current_seed_finished = True
        # elif self.__is_discovery(message):
        #     super().on_progress(seed_number)
        elif self.__is_saved(message):
            super().on_save(seed_number, self._get_current_checkpoint_id(seed_number))     
        elif self.__is_error(message):
            super().on_error(seed_number, self._get_current_checkpoint_id(seed_number))
            is_current_seed_finished = True
        
        if is_current_seed_finished:
            self.threading_lock.acquire()
            self.nb_seeds_finished += 1
            self.threading_lock.release()
        
        return logger_name, log_level_name, seed_number, log_id, message

    def test_file_exist(self, file_path):
        self.shell.prompt(timeout=2)
        self.shell.sendline('test -f '+file_path + '&& echo "File exist" || echo "File does not exist"')
        self.shell.sendline('echo "test_file_end"')
        self.shell.expect('test_file_end')
        lines = self.shell.before.decode().split("\n")
        for line in lines:
            if "File exist" == line.strip():
                return True
            if "File does not exist" == line.strip():
                return False
        return False

    def __listen_log_file(self, local_folder):
        """
        Brief: As long as the seeds have not finished read log file and do appropriate action
        """
        try:
            current_log_line = previous_log = None
            previous_log_level_name = previous_seed_number = previous_message = None
            log_level_name = seed_number = message = None
            previous_checkpoint_ids = []
            while self.nb_seeds_finished < self.experiment_config["experiment"]["config"]["nb_seeds"]:
                self.shell.expect('\n')
                current_log_line = self.shell.before.decode()
                if "ad_tool_logger -" in current_log_line:
                    ## determine action to do
                    logger_name, log_level_name, seed_number, log_id, message = self.__parse_log(copy(current_log_line), local_folder)
                    
                    if self.__log_not_in_db(log_id):
                        previous_checkpoint_ids.append(self._get_current_checkpoint_id(seed_number))
                        if previous_log is not None and previous_message is not None:
                            self.app_db_logger_handler.save(
                                previous_checkpoint_ids.pop(0), 
                                previous_seed_number, 
                                self.app_db_logger_handler.log_levels_id[previous_log_level_name],
                                log_id,
                                previous_message)
                            
                        previous_log = current_log_line
                        previous_log_level_name = log_level_name
                        previous_seed_number = seed_number
                        previous_message = message
                else:
                    #aggregates the text to the next log we will send to DB
                    if previous_log is not None and previous_message is not None:
                        previous_message += current_log_line
                    elif message is not None:
                        previous_log = self.shell.before.decode()
                        previous_message = message
            if "[ERROR]" in previous_message:
                #In this case we leave while with only a part of the message
                self.shell.prompt()
                error_message = self.shell.before.decode()
                #get all text and keep only the error
                previous_message = error_message[error_message.find(previous_message):]
            #send last log
            self.app_db_logger_handler.save(
                previous_checkpoint_ids.pop(0), 
                previous_seed_number, 
                self.app_db_logger_handler.log_levels_id[previous_log_level_name],
                log_id,
                previous_message)
            ## close properly ssh connection
            self.shell.close()
        except Exception as ex:
            print("unexpected error occurred. checked the logs of your remote server")
            print(ex)
            self.callback_to_all_running_seeds(lambda seed, current_checkpoint_id : super().on_error(seed, current_checkpoint_id))
        
    def __is_finished(self, log):
        return match_except_number(log.strip(), "- [FINISHED] - experiment 0 with seed 0 finished")
               
    # def __is_discovery(self, log):
    #     return match_except_number(log.strip(), "- [DISCOVERY] - New discovery from experiment 0 with seed 0")
    
    def __is_saved(self, log):
        return match_except_number(log.strip(), "- [SAVED] - experiment 0 with seed 0 saved")

    def __is_error(self, log):
        return '[ERROR]' in log
    
    def __log_not_in_db(self, log_name):
        response = self._app_db_caller("/logs?&name=eq.{}".format(log_name), AppDBMethods.GET, None)
        return response.content.decode() == '[]'

    def __new_files_saved_on_remote_disk(self, log):
        return "New discovery saved" in log or "New modules saved"  in log
    
    def __get_log_part(self, log):
        log_splitted = log.split("-")
        logger_name = log_splitted[0].strip()
        log_level_name = log_splitted[1].strip()
        seed_number = log_splitted[2].replace("SEED", "").strip()
        log_id = log_splitted[3].replace("LOG_ID", "").strip()
        message = ""
        for i in range(4, len(log_splitted)):
            message += "-" + log_splitted[i]
        return logger_name, log_level_name, int(seed_number), log_id, message

    def __get_saved_files_to_pull(self, log):
        splitted_log = log.split(':')
        del splitted_log[0]
        path = splitted_log[0].strip()
        run_idx = splitted_log[2].strip()
        sub_folders = splitted_log[1].replace('[', '')
        sub_folders = sub_folders.replace(']', '')
        sub_folders = sub_folders.replace('\'', '')
        sub_folders = sub_folders.replace(' ', '')
        sub_folders = sub_folders.split(',')
        if '' in sub_folders:
            sub_folders = sub_folders.remove('')
        return path, sub_folders, int(run_idx)

    def __get_run_id(self):
        self.shell_to_get_run_id = pxssh.pxssh()
        self.shell_to_get_run_id.login(self.__host_profile["ssh_configuration"], ssh_config=self.ssh_config_file_path)

        while not self.test_file_exist(self.__host_profile["work_path"]+"/run_ids/{}".format(self.id)):
            self._app_db_caller("/preparing_logs", 
                                AppDBMethods.POST, {
                                    "experiment_id":self.id,
                                    "message": "waiting for the server"
                                }
                            )
            sleep(self.__host_profile["check_experiment_launched_every"])

        while not "[RUN_ID_start]" in self.shell_to_get_run_id.before.decode():
            self.shell_to_get_run_id.prompt()
            self.shell_to_get_run_id.sendline('cat '+self.__host_profile["work_path"]+"/run_ids/{}".format(self.id))

        lines = self.shell_to_get_run_id.before.decode().split("\n")
        self.shell_to_get_run_id.close()
        
        for line in lines:
            if line.startswith("[RUN_ID_start]") and "[RUN_ID_stop]" in line:
                line = line.replace("RUN_ID_stop", "")
                line = line.replace("RUN_ID_start", "")
                line = line.replace("[", "")
                line = line.replace("]", "")
                line = line.strip()
                response = self._app_db_caller("/experiments?id=eq.{}".format(self.id), 
                                AppDBMethods.PATCH, 
                                {"remote_run_id": line})
                return line

#endregion

#region save to db
    def save_discovery_to_expe_db(self, **kwargs):
            """
            brief:      callback saves the discoveries outputs we want to save on database.
            comment:    always saved : run_idx(json), experiment_id(json)
                        saved if key in self.to_save_outputs: raw_run_parameters(json)
                                                            run_parameters,(json)
                                                            raw_output(file),
                                                            output(json),
                                                            rendered_output(file),
                                                            step_observations(file)
            """
            try:
                saves={}
                files_to_save={}
                to_save_outputs = copy(kwargs["sub_folders"])
                if to_save_outputs is not None:
                    to_save_outputs.extend(["run_idx", "experiment_id", "seed"])
                else:
                    to_save_outputs = ["run_idx", "experiment_id", "seed"]
                folder = "{}/outputs/{}/{}/".format(kwargs["folder"],self.id,  kwargs["seed"])

                for save_item in to_save_outputs:
                    if kwargs["sub_folders"] is not None and save_item in kwargs["sub_folders"]:
                        # if save_item == "step_observations":
                        #     kwargs[save_item] = serialize_autodisc_space(kwargs[save_item])

                        if save_item == "raw_output" or save_item == "step_observations":
                            files_to_save[save_item] = ('{}_{}_{}'.format(
                                save_item, kwargs["experiment_id"], kwargs["run_idx"]), 
                                open(folder+save_item+"/idx_{}.pickle".format(kwargs["run_idx"]), "rb"))
                        elif save_item == "rendered_output":
                            extension = os.listdir(folder+save_item)[0].split(".")[1]
                            files_to_save[save_item] = open(folder+save_item+"/idx_{}.{}".format(kwargs["run_idx"], extension), "rb")
                        else:
                            saves[save_item] = serialize_autodisc_space(pickle.load(open(folder+save_item+"/idx_{}.pickle".format(kwargs["run_idx"]), "rb")))
                    else:
                        saves[save_item] = serialize_autodisc_space(kwargs[save_item])
                discovery_id = self._expe_db_caller("/discoveries", request_dict=saves)["ID"]
                #check dict files_to_saves is empty or not
                if files_to_save:
                    self._expe_db_caller("/discoveries/" + discovery_id + "/files", files=files_to_save)
            except Exception as ex:
                print("ERROR : error while saving discoveries in experiment {} run_idx {} seed {} = {}".format(self.id, kwargs["run_idx"], kwargs["seed"], traceback.format_exc()))
    
    def save_modules_to_expe_db(self, **kwargs):
        try:
            to_save_modules = kwargs["sub_folders"]
            folder = "{}/checkpoints/{}/{}/".format(kwargs["folder"],self.id,  kwargs["seed"])

            files_to_save={} 
            for module in to_save_modules:
                files_to_save[module] = open(folder+module+"/idx_{}.pickle".format(kwargs["run_idx"]), "rb")
            
            module_id = self._expe_db_caller("/checkpoint_saves", 
                                        request_dict={
                                            "checkpoint_id": self._get_current_checkpoint_id(kwargs["seed"]),
                                            "run_idx": kwargs["run_idx"],
                                            "seed": kwargs["seed"]
                                        }
                                    )["ID"]
            self._expe_db_caller("/checkpoint_saves/" + module_id + "/files", files=files_to_save)
        except Exception as ex:
            print("ERROR : error while saving modules in experiment {} run_idx {} seed {} = {}".format(self.id, kwargs["run_idx"], kwargs["seed"], traceback.format_exc()))
#endregion