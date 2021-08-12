import { Component, OnInit, AfterViewInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/app-db.service';
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
  public progressPercent:number = 0;
  private intervalToSubscribe: Observable<number> | undefined;
  private updateSubscription: Subscription | undefined;
  public autoRefreshSeconds: number = 30;

  public jupyterSession:JupyterSessions[] = [];
  public actual_session_path : string | undefined;
  public new_data_available_sub: Subscription | undefined;
  public new_data_available: boolean = false;
  public jupy_session_find = false;
  public kernelInfoSet =false;
  public newDiscoveriesExist = false;

  public kernel_restart_sub: Subscription | undefined;

  public message = 'import requests'+ '\n' +
                   'import json'+ '\n' +
                   'experiment = json.loads(requests.get(url = "http://127.0.0.1:3000/experiments").content.decode())'

  
  objectKeys = Object.keys;
  urlSafe: SafeResourceUrl | undefined;
  
  constructor(private appDBService: AppDbService, private AutoDiscServerService: AutoDiscServerService, private route: ActivatedRoute,
              public sanitizer: DomSanitizer, private jupyterService: JupyterService) { }

  ngOnInit() {
    this.refreshExperiment();
    this.resetAutoRefresh();

    this.initJupyterServiceSubData();
    this.initPopover();    
  }

  

  resetAutoRefresh(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
    this.intervalToSubscribe = interval(this.autoRefreshSeconds*1000);
    this.updateSubscription = this.intervalToSubscribe.subscribe(
      (val) => { this.refreshExperiment()});
  }

  refreshExperiment(): void {
    this.appDBService.getExperimentById(
      Number(this.route.snapshot.paramMap.get('id'))
    )
    .subscribe(experiment => {
      this.newDiscoveriesExist = (this.newDiscoveriesExist || this.experiment == undefined || this.experiment.progress < experiment.progress);
      this.experiment = experiment;
      this.progressPercent = this.experiment.progress/experiment.config.nb_iterations*100;
      if (experiment.exp_status == 1){
        this.ellapsed = ((new Date().getTime() - new Date(experiment.created_on).getTime()) / 1000 / 60 / 60).toFixed(2);
      }

      if(!this.kernelInfoSet){
        this.defineInfoToAccessKernel(this.experiment.name, this.experiment.id);
      }
      if(!this.jupy_session_find){
        this.getJupyterSessions();
      }

      if(this.newDiscoveriesExist && this.jupy_session_find){      
        this.jupyterService.sendToKernel(this.message);
        this.newDiscoveriesExist = false;
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
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
    this.jupyterService.closeKernelChannel()
  }

  initPopover(){
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
      return new bootstrap.Popover(popoverTriggerEl)
    })
  }

  // ######### JUPYTER PART #########

  defineInfoToAccessKernel(exp_name: string, exp_id: number){
    this.urlSafe= this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8888/notebooks/'+exp_name+'_'+exp_id.toString()+'.ipynb');
    // this.urlSafe= this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8888/lab/Untitled_Folder');
    // this.urlSafe= this.sanitizer.bypassSecurityTrustResourceUrl('http://localhost:8888/lab?');
    this.actual_session_path = exp_name+'_'+ exp_id.toString()+'.ipynb';
    this.kernelInfoSet = true;
  }

  getJupyterSessions():void{
    this.jupyterService.getSessions()
    .subscribe(jupyterSession => {this.jupyterSession = jupyterSession,
      this.openKernelChannel()     
    });
  }

  openKernelChannel():any{
    for (let session of this.jupyterSession) {
      if(session.path == this.actual_session_path){
        this.jupy_session_find = true;
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
    // when the kernel was restart this reconnect our app to the good one.
    this.kernel_restart_sub = this.jupyterService.kernel_restart.subscribe((kernel_restart: boolean) => {
      if(this.experiment && kernel_restart){
        this.newDiscoveriesExist = true;
        this.getJupyterSessions();
      }    
    });
  }

}
