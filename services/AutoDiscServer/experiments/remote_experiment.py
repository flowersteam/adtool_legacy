from AutoDiscServer.experiments import BaseExperiment
from AutoDiscServer.utils import list_profiles, parse_profile
import socket
import os
from ssh2.session import Session

class RemoteExperiment(BaseExperiment):
    def __init__(self, host_profile_name, *args, **kwargs):

        # super().__init__(*args, **kwargs)
        
        # self.experiment_config['callbacks']['on_discovery'][0]['name'] = 'expeDB'
        # self.experiment_config['callbacks']['on_discovery'][0]['config']['base_url'] = 'http://127.0.0.1:5001'

        # self.experiment_config['callbacks']['on_saved'][0]['name'] = 'expeDB'
        # self.experiment_config['callbacks']['on_saved'][0]['config']['base_url'] = 'http://127.0.0.1:5001'

        additional_callbacks = {
            "on_discovery": [],
            "on_save_finished": [],
            "on_finished": [],
            "on_cancelled": [],
            "on_error": [],
            "on_saved": [],
        }

        self.__host_profile = parse_profile(next(profile[1] for profile in list_profiles() if profile[0] == host_profile_name))
        self.port = 22

        #create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.__host_profile["ssh_configuration"], self.port))

        #create session and connect
        self.session = Session()
        self.session.handshake(sock)
        self.session.agent_auth(user) # TODO
        self.channel = self.session.open_session()
        # create shell
        self.channel.shell()
        # shell to listen logs in real time
        self.channel2 = self.session.open_session()
        self.channel2.shell()


        print("test")
        #TODO: Package everything and put it on server 
        #TODO: Write PID in DB!!

    def start(self):
        #TODO: Launch + monitor
        raise NotImplementedError()

    def stop(self):
        #TODO: call cancel with PID
        raise NotImplementedError()

    def _monitor(self):
        '''
            Check status + call callbacks
        '''
        raise NotImplementedError()

    def __tar_local_folder(self, folder_src, folder_dest):
        folder_src_path_to = folder_src.split("/")
        target_folder = folder_src_path_to[len(folder_src_path_to)-1]
        index = folder_src.rfind(target_folder)
        folder_src = folder_src[:index]
        os.system('cd {} && tar -czvf {} {}'.format(folder_src, folder_dest, target_folder))

    def __push_folder(self, folder_path_local, folder_path_remote):
        os.system('scp -r -P {} {} {}:{}'.format(self.port, folder_path_local, self.user+"@"+self.host, folder_path_remote))
    
    def __pull_folder(self, folder_path_remote, folder_path_local):
        os.system('scp -r -P {} {}:{} {}'.format(self.port, self.user+"@"+self.host, folder_path_remote, folder_path_local))

    def __untar_folder(self, folder_path_src, folder_path_dest):
        self.channel.write('tar -xvf {} -C {} \n'.format(folder_path_src, folder_path_dest))
    
    def __exec_python(self, file_path, config_file_path, exp_id, seed):
        """
        normalement avec curta tout doit être gérer par le sbatch donc juste besoin du chemin vers lui (pas des autres arguments)
        et comme seule ligne de code dans la fonction: self.channel.write('sbath {}/testbatch.slurm'.format(file_path))
        """
        self.channel.write('bash -i \n') # docker
        self.channel.write('export PYTHONPATH="${PYTHONPATH}:/home/auto_disc/libs/" \n')# docker
        self.channel.write('export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/lib/ \n')# docker
        self.channel.write('conda activate autoDiscTool  >> /home/auto_disc/out 2>> /home/auto_disc/error \n')# docker
        self.channel.write('mkdir /home/auto_disc/logs  >> /home/auto_disc/out 2>> /home/auto_disc/error \n') # docker
        self.channel.write('python {} --config_file {} --experiment_id {} --seed {} \n'.format(file_path, config_file_path, exp_id, seed))# docker
        
        #self.channel.write('sbath {}/testbatch.slurm \n'.format(file_path))# curta




    # def on_progress(self, **kwargs):
    #     super().on_progress(kwargs["seed"])

    # def on_save(self, **kwargs):
    #     self.threading_lock.acquire()
    #     res = super().on_save(kwargs["seed"], kwargs["checkpoint_id"])
    #     self.threading_lock.release()
    #     return {"checkpoint_id": res}

    # def on_error(self, **kwargs):
    #     self.threading_lock.acquire()
    #     res =  super().on_error(kwargs["seed"], kwargs["checkpoint_id"], kwargs["message"])
    #     self.threading_lock.release()
    #     return res

    # def on_finished(self, **kwargs):
    #     self.threading_lock.acquire()
    #     super().on_finished(kwargs["seed"])
    #     self.threading_lock.release()
    
    # def on_cancelled(self, **kwargs):
    #     self.threading_lock.acquire()
    #     super().on_cancelled(kwargs["seed"])
    #     self.threading_lock.release()