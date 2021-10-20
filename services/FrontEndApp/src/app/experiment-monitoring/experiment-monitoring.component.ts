import { Component, OnInit, AfterViewInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/app-db.service';
import { ExpeDbService } from '../services/expe-db.service';
import { AutoDiscServerService } from '../services/auto-disc.service';
import { Experiment } from '../entities/experiment';
import { Observable, interval, Subscription, empty } from 'rxjs';

import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { JupyterService } from '../services/jupyter.service';
import { JupyterSessions } from '../entities/jupyter_sessions';

import  * as bootstrap  from 'bootstrap'

@Component({
  selector: 'app-experiment-monitoring',
  templateUrl: './experiment-monitoring.component.html',
  styleUrls: ['./experiment-monitoring.component.scss']
})
export class ExperimentMonitoringComponent implements OnInit {

  experiment: Experiment | undefined;
  public ellapsed: string | undefined;
  public progressPercent:string = "0";
  private intervalToSubscribe: Observable<number> | undefined;
  private updateSubscription: Subscription | undefined;

  // var about jupyter
  public jupyterSession:JupyterSessions[] = [];
  public actual_session_path : string | undefined;
  public new_data_available_sub: Subscription | undefined;
  public new_data_available: boolean = false;
  public kernelInfoSet =false;
  public message:string = '';
  aKernelToRuleThemAll: any;
  needSendMessageToKernel = true;

  tabButtonDisable:any; //able or disbale button to select page (jupyter Discovery Log)

  // var about Discovery display
  actualRunIdx:any = []; // run_idx value defined by the user. Using to show discovery of this specific run_idx
  all_seed_checkox_selected:any = []; // seed value defined by the user to visualise it in discoveries tab
  nb_discoveries_display = 12; // how many discories we want display simultaneously
  index_discoveries_display:number=0; //index to define wich subarray of run_idx we want display now
  array_filter_run_idx:any = []; //subarray of run_idx we want see now
  slider_double_value = {
    value: 0,
    highValue: 0,
    options: {
      floor: 0,
      ceil: 0
   } // define var needed to range with two values
  }

  // var about logs
  logs ={
    checkpoints :[],
    seeds :[],
    log_level :<any>[],
  };
  all_checkpoints_logs_checkox_selected: any = [];
  all_seed_checkox_logs_selected: any = [];
  all_level_checkox_logs_selected: any = [];
  logs_value :any = [];
  

  
  public autoRefreshSeconds: number = 5;
  objectKeys = Object.keys;
  urlSafe: SafeResourceUrl | undefined;
  
  constructor(private appDBService: AppDbService, private AutoDiscServerService: AutoDiscServerService, private route: ActivatedRoute,
              public sanitizer: DomSanitizer, private jupyterService: JupyterService, private expeDbService: ExpeDbService) { }

  ngOnInit() {
    this.refreshExperiment();
    this.resetAutoRefresh();
    this.jupyterService.createKernel().subscribe(res =>{
      this.aKernelToRuleThemAll = res;
      this.jupyterService.openKernelChannel(this.aKernelToRuleThemAll.id).subscribe(_ =>{
        if(this.needSendMessageToKernel){
          this.makeKernelMessageToCreateDataset();
          this.jupyterService.sendToKernel(this.message);
          this.needSendMessageToKernel = false;
        }
      });
    });
    
    this.initPopover();
    this.initCollapseVisualisation();
    this.getLogLevels()
  }

  

  resetAutoRefresh(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
    if (this.experiment?.exp_status == 1){
      this.intervalToSubscribe = interval(this.autoRefreshSeconds*1000);
      this.updateSubscription = this.intervalToSubscribe.subscribe(
        (val) => { this.refreshExperiment()});
    }
  }

  get refreshExperimentMethod() {
    return this.refreshExperiment.bind(this);
  }

  refreshExperiment(): void {
    this.appDBService.getExperimentById(
      Number(this.route.snapshot.paramMap.get('id'))
    )
    .subscribe(experiment => {
      this.experiment = experiment;
      this.progressPercent = (this.experiment.progress/experiment.config.nb_iterations*100).toFixed(1);
      this.experiment.checkpoints.sort((a, b) => {return a.id - b.id})
      if (experiment.exp_status == 1){
        this.ellapsed = ((new Date().getTime() - new Date(experiment.created_on).getTime()) / 1000 / 60 / 60).toFixed(2);
      }

      this.defineWhatWeWantVisualise();

      if(!this.kernelInfoSet){
        this.defineInfoToAccessKernel(this.experiment.name, this.experiment.id);
      }

      this.slider_double_value = {
        value: this.slider_double_value.value,
        highValue: this.slider_double_value.highValue,
        options: {
          floor: 0,
          ceil: this.experiment.progress-1
       }
      }
    });
  }

  stopExperiment(): void {
    if (this.experiment != undefined){
      console.log("Stoppping experiment with id " + this.experiment.id)
      this.AutoDiscServerService.stopExperiment(this.experiment.id).subscribe(
        (val) => {this.refreshExperiment();}
      )
    }
  }

  ngOnDestroy(): void{
    this.jupyterService.destroyKernel(this.aKernelToRuleThemAll.id).subscribe();
    this.jupyterService.closeKernelChannel();
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;    
  }

  initPopover(){
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
      return new bootstrap.Popover(popoverTriggerEl)
    })
  }

  // ######### TAB PART (jupyter discovery log)#########
  initCollapseVisualisation(){
    this.tabButtonDisable = {"btncollapseJupyter": true, "btncollapseDiscovery": false, "btncollapseLogs": false};
  }

  collapseVisualisation(event: any){
    
    // var collapseBtnTriggerList = document.querySelectorAll('#btncollapseJupyter, #btncollapseDiscovery, #btncollapseLogs')
    var collapseTriggerList:any = [].slice.call(document.querySelectorAll('#collapseJupyter, #collapseDiscovery, #collapseLogs'))
    var collapseBtnTriggeringList =[];
    for(let collapseTrigger of collapseTriggerList){
      if(event.delegateTarget.id.replace("btn", "") == collapseTrigger.id || collapseTrigger.classList.value.includes('show')){
        collapseBtnTriggeringList.push(collapseTrigger);
      }
    }
     
    var collapseList = collapseBtnTriggeringList.map(function (collapseTriggerEl) {
      return new bootstrap.Collapse(collapseTriggerEl)
    })

    for (let key in this.tabButtonDisable) {
      this.tabButtonDisable[key] = false;
    }
    this.tabButtonDisable[event.delegateTarget.id] = true;
  }

  // ######### DISCOVERY PART #########

  counter(i: number) {
    let res = new Array(i);
    for (let index = 0; index < res.length; index++) {
      res[index] = index;
      
    }    
    return res;
}

  setActualRunIdx(){
    this.actualRunIdx = [];
    for (let index = this.slider_double_value.value; index <= this.slider_double_value.highValue; index++) {
        this.actualRunIdx.push(index)
    }
  }

  setActualCheckbox(checkbox_name: string){   
    if(checkbox_name == "seed_checkbox"){
      this.all_seed_checkox_selected = [];
    }
    else if(checkbox_name == "seed_checkbox_logs"){
      this.all_seed_checkox_logs_selected = [];
    }
    else if(checkbox_name == "checkpoint_checkbox_logs"){
      this.all_checkpoints_logs_checkox_selected = [];
    }
    else if(checkbox_name == "level_checkbox_logs"){
      this.all_level_checkox_logs_selected = [];
    }
    let checkbox = document.querySelectorAll('input[name="'+checkbox_name+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      if((checkbox[index]as HTMLInputElement).checked){
        if(checkbox_name == "seed_checkbox"){
          let res = checkbox[index].id.replace('checkboxSeed+','');
          this.all_seed_checkox_selected.push(parseInt(res))
        }
        else if(checkbox_name == "seed_checkbox_logs"){
          let res = checkbox[index].id.replace('checkboxSeedLogs+','');
          this.all_seed_checkox_logs_selected.push(parseInt(res))
        }
        else if(checkbox_name == "checkpoint_checkbox_logs" && this.experiment){
          let res = checkbox[index].id.replace('checkboxCheckpointLogs+','');
          this.all_checkpoints_logs_checkox_selected.push(this.experiment.checkpoints[parseInt(res)].id)
        }
        else if(checkbox_name == "level_checkbox_logs"){
          let res = checkbox[index].id.replace('checkboxLevelLogs+','');
          this.all_level_checkox_logs_selected.push(parseInt(res))
        }
        
      } 
    }
  }

  selectAllCheckbox(event: any){
    let checkbox = document.querySelectorAll('input[name="'+event.target.name+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      (checkbox[index]as HTMLInputElement).checked = true;
    }
    this.setActualCheckbox(event.target.name);
  }

  unselectAllCheckbox(event: any){
    let checkbox = document.querySelectorAll('input[name="'+event.target.name+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      (checkbox[index]as HTMLInputElement).checked = false;
    }
    this.setActualCheckbox(event.target.name);
  }

  defineWhatWeWantVisualise(){
    this.index_discoveries_display = 0
    this.setActualRunIdx();
    this.setActualCheckbox("seed_checkbox");
    this.getDiscovery();
  }

  definedFilters(): string{
    let filter = "";
    if(this.experiment){
      this.array_filter_run_idx = [];
      for (let i = 0; i <= Math.floor(this.actualRunIdx.length / this.nb_discoveries_display); i++) {
        this.array_filter_run_idx.push(this.actualRunIdx.slice(i*this.nb_discoveries_display, (i+1)*this.nb_discoveries_display))
        if(this.array_filter_run_idx[i].length == 0){
          this.array_filter_run_idx.splice(i, 1); 
        }
      }
      filter = '{"$and":[{"experiment_id":'
                    +this.experiment.id.toString()
                    +'}, {"run_idx":{"$in":'
                    +JSON.stringify(this.array_filter_run_idx[this.index_discoveries_display])
                    +'}},  {"seed":{"$in":'
                    +JSON.stringify(this.all_seed_checkox_selected)
                    +'}}]}'
    }
    return filter;
  }
  
  setIndexDiscoveries(i:number){
    this.index_discoveries_display = this.index_discoveries_display + i;
    if(i == 0){
      this.index_discoveries_display = 0;
    }
    else if(this.index_discoveries_display < 0){
      this.index_discoveries_display =this.array_filter_run_idx.length -1;
    }
    else if(this.index_discoveries_display >= this.array_filter_run_idx.length){
      this.index_discoveries_display = 0;
    }
    this.getDiscovery();
  }


  getDiscovery(): void {
    if(this.experiment){
      for(let index = 0; index < this.experiment.config.nb_seeds; index++){
          let video = <HTMLVideoElement><any> document.querySelector("#video_"+index.toString());
          if(video){
            video.src = "";
          }
      }
      let filter = this.definedFilters()
      this.expeDbService.getDiscovery(filter)
      .subscribe(discoveries => {
        if(discoveries.length > 0){
          for(let discoverie of discoveries){
            this.expeDbService.getDiscoveryRenderedOutput(discoverie._id)
            .subscribe(renderedOutput => {
              let video = <HTMLVideoElement> <any> document.querySelector("#video_"+discoverie.seed.toString()+"_"+discoverie.run_idx.toString());
              if(video){
                video.src = window.URL.createObjectURL(renderedOutput);
              } 
            });
          }
          
        }
        
      });
    }
  }

  // ######### JUPYTER PART #########

  defineInfoToAccessKernel(exp_name: string, exp_id: number){
    let path = exp_name+'_'+exp_id.toString()
    this.urlSafe= this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8888/lab/workspaces/'+path+'/tree/Experiments/'+path+'/');
    this.actual_session_path = exp_name+'_'+ exp_id.toString()+'.ipynb';
    this.kernelInfoSet = true;
  }

  openKernelChannel():any{
    for (let session of this.jupyterSession) {
      if(session.path == this.actual_session_path){
        this.jupyterService.openKernelChannel(session.kernel.id.toString());
        this.refreshExperiment();
        break;
      } 
    }
  }

  initJupyterServiceSubData(){
    //set new_data_available use to show (or not) the notification and his popover. To say at user new data available
    this.new_data_available_sub = this.jupyterService.new_data_available.subscribe((new_data_available: boolean) => {
      this.new_data_available = new_data_available;
    });
  }

  makeKernelMessageToCreateDataset(){
    if(this.experiment){
      this.message =  'import os'+'\n' +
                      'import sys'+'\n' +
                      'server_path = os.readlink("/proc/%s/cwd" % os.environ["JPY_PARENT_PID"])'+'\n' +
                      'module_path = os.path.abspath(os.path.join(server_path, "../../libs"))'+'\n' +
                      'if module_path not in sys.path:'+'\n' +
                      '    sys.path.append(module_path)'+'\n' +
                      'from auto_disc_db import Dataset'+'\n' +
                      'if __name__ == "__main__":'+'\n' +
                      '     dataset_'+this.experiment.id.toString()+' = Dataset('+this.experiment.id.toString()+')'+'\n' +
                      '     dataset = Dataset('+this.experiment.id.toString()+')'+'\n' +
                      '     print(dataset)'+'\n'+
                      '     %store dataset_'+this.experiment.id.toString()+'\n' +
                      '     %store dataset'
                      
      }
  }


  // ######### LOGS PART #########
  setLogs(){
    this.logs.checkpoints = []
    this.logs.seeds = []
    this.logs.log_level = []
  }

  getLogLevels(){
    this.appDBService.getAllLogLevels().subscribe(res =>{ this.logs.log_level = res});
  }

  definedOneFilterParam(param:string, param_name:string){
    param = param.replace("[", "(");
    param = param.replace("]", ")");
    if(param.length <= 2){
      param = ""
    }
    else{
      param = "&"+param_name+"=in."+param
    }
    return(param)
  }

  collapseLogs(event: any){
    var collapseTriggerList:any = [].slice.call(document.querySelectorAll('#collapseCheckBoxSeedLogs, #collapseCheckBoxCheckpointLogs, #collapseCheckBoxLevelLogs'))
    var collapseBtnTriggeringList =[];
    for(let collapseTrigger of collapseTriggerList){
      if(event.delegateTarget.id.replace("btn", "") == collapseTrigger.id || collapseTrigger.classList.value.includes('show')){
        collapseBtnTriggeringList.push(collapseTrigger);
      }
    }
     
    var collapseList = collapseBtnTriggeringList.map(function (collapseTriggerEl) {
      return new bootstrap.Collapse(collapseTriggerEl)
    })
  }

  logsWewant(){
    if(this.experiment){
      this.setActualCheckbox("seed_checkbox_logs");
      this.setActualCheckbox("checkpoint_checkbox_logs");
      this.setActualCheckbox("level_checkbox_logs");
      let checkpoints = this.definedOneFilterParam(JSON.stringify(this.all_checkpoints_logs_checkox_selected), 'checkpoint_id');
      let seeds = this.definedOneFilterParam(JSON.stringify(this.all_seed_checkox_logs_selected), 'seed');
      let log_levels = this.definedOneFilterParam(JSON.stringify(this.all_level_checkox_logs_selected), 'log_level_id');
      let filter = "?&experiment_id=eq."+this.experiment.id.toString() + checkpoints + log_levels + seeds;
      this.appDBService.getLogs(filter).subscribe(res => this.logs_value = res);
      console.log(filter);
    }
  }
}
