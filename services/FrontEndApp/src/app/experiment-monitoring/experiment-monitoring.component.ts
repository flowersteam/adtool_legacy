import { Component, OnInit, AfterViewInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/app-db.service';
import { ExpeDbService } from '../services/expe-db.service';
import { AutoDiscServerService } from '../services/auto-disc.service';
import { Experiment } from '../entities/experiment';
import { Observable, interval, Subscription } from 'rxjs';

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

  // var about Discovery display
  actualRunIdx = 0; // run_idx value defined by the user. Using to show discovery of this specific run_idx
  run_idxmax= 0; // define max value of range who permit to user to chose run_idx he want to see
  actualSeed=0; // seed value defined by the user to visualise it in discoveries tab
  discoveries:any;
  renderedOutput:any;
  tabButtonDisable:any;

  
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
        this.makeKernelMessageToCreateDataset();
        this.jupyterService.sendToKernel(this.message);
      });
    });
    
    this.initPopover();
    this.initCollapseVisualisation();  
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
      this.run_idxmax = this.getUpToNModEqualY(experiment.config.save_frequency, 0, experiment.config.nb_iterations);
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

      if(this.aKernelToRuleThemAll){
        this.makeKernelMessageToCreateDataset()     
        this.jupyterService.sendToKernel(this.message);
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

  // ######### TAB PART (jupyter discovery...)#########
  initCollapseVisualisation(){
    this.tabButtonDisable = {"btncollapseJupyter": true, "btncollapseDiscovery": false};
  }

  collapseVisualisation(event: any){
    
    var collapseBtnTriggerList = document.querySelectorAll('#btncollapseJupyter, #btncollapseDiscovery')
    var collapseTriggerList = [].slice.call(document.querySelectorAll('#collapseJupyter, #collapseDiscovery'))
    var collapseList = collapseTriggerList.map(function (collapseTriggerEl) {
      return new bootstrap.Collapse(collapseTriggerEl)
    })

    for (let key in this.tabButtonDisable) {
      this.tabButtonDisable[key] = false;
    }
    this.tabButtonDisable[event.delegateTarget.id] = true;
  }

  // ######### DISCOVERY PART #########
  getUpToNModEqualY(x: number, y: number, n: number){
    // Stores the required number
    let num = 0;
 
    //  Update num as the result
    if (n - n % x + y <= n){
      num = n - n % x + y;
    }
    else{
      num = n - n % x - (x - y);
    }
    return num;
  }

  counter(i: number) {
    return new Array(i);
}

  setActualRunIdx(event: any){
    this.actualRunIdx = event.target.value;
    console.log(this.actualRunIdx);
    this.defineWhatWeWantVisualise()
  }

  setActualSeedToVisualise(event: any){
    this.actualSeed = event.target.value;
    this.defineWhatWeWantVisualise();
  }

  defineWhatWeWantVisualise(){
    // this.actualRunIdx
    // this.actualSeed
    this.getDiscovery();
  }

  getDiscovery(): void {
    if(this.experiment){
      let filter = '{"$and":[{"experiment_id":{"$eq":'+this.experiment.id.toString()+'}}, {"run_idx":{"$eq":'+this.actualRunIdx.toString()+'}},  {"seed":{"$in":[0,1,2]}}]}'
      this.expeDbService.getDiscovery(filter)
      .subscribe(discoveries => {
        this.discoveries = discoveries;
        if(discoveries.length > 0){
          this.expeDbService.getDiscoveryRenderedOutput(this.discoveries[0]._id)
          .subscribe(renderedOutput => {
            this.renderedOutput = renderedOutput;
            let video = document.querySelector("video");
            if(video){
              video.src = window.URL.createObjectURL(this.renderedOutput);
            }     
          });
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

}
