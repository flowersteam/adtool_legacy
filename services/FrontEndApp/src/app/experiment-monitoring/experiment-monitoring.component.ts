import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/REST-services/app-db.service';
import { AutoDiscServerService } from '../services/REST-services/auto-disc.service';
import { ToasterService } from '../services/toaster.service';
import { Experiment } from '../entities/experiment';
import { Observable, interval, Subscription, empty } from 'rxjs';

import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { RESTResponse } from '../entities/rest_response';

@Component({
  selector: 'app-experiment-monitoring',
  templateUrl: './experiment-monitoring.component.html',
  styleUrls: ['./experiment-monitoring.component.scss'],
})
export class ExperimentMonitoringComponent implements OnInit {

  experiment: Experiment | undefined;
  public ellapsed: string | undefined;
  public progressPercent:string = "0";
  public autoRefreshSeconds: number = 5;
  public allowCancelButton: boolean = true;

  private intervalToSubscribe: Observable<number> | undefined;
  private updateSubscription: Subscription | undefined;

  objectKeys = Object.keys;
  urlSafe: SafeResourceUrl | undefined;
  
  constructor(private appDBService: AppDbService, private AutoDiscServerService: AutoDiscServerService, private route: ActivatedRoute,
              public sanitizer: DomSanitizer, private toasterService: ToasterService) { }

  ngOnInit() {
    this.resetAutoRefresh();
  }

  resetAutoRefresh(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
    if (this.experiment == null || this.experiment?.exp_status == 1 || this.experiment?.exp_status == 4){
      this.refreshExperiment();
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
    .subscribe(response => {
      if(response.success && response.data){
        this.experiment = response.data;
        this.progressPercent = (this.experiment.progress/this.experiment.config.nb_iterations*100).toFixed(1);
        this.experiment.checkpoints.sort((a, b) => {return a.id - b.id})
        if(this.experiment.exp_status == 1){
          this.ellapsed = ((new Date().getTime() - new Date(this.experiment.created_on).getTime()) / 1000 / 60 / 60).toFixed(2);
        }
        else{
          this.resetAutoRefresh();
        }
      }
      else {
        this.toasterService.showError(response.message ?? '', "Error refreshing experiment");
      }
    });
  }

  get callObservableStopExperimentMethod() {
    return this.callObservableStopExperiment.bind(this);
  }

  stopExperiment(): void {
    this.callObservableStopExperiment().subscribe(
      response => {
        if(!response.success){
          this.toasterService.showError(response.message ?? '', "Error stopping experiment", {timeOut: 0, extendedTimeOut: 0});
          this.toasterService.showWarning("Experiment is considered cancelled but may still run, please consider checking host.", "Experiment cancellation has failed", {timeOut: 0, extendedTimeOut: 0})
        }

        this.refreshExperiment();
        this.allowCancelButton = true;
      }
    )
  }

  callObservableStopExperiment(): Observable<RESTResponse<any>> {
    if (this.experiment != undefined){
      this.toasterService.showInfo("Cancelling experiment...", "Cancel");
      this.allowCancelButton = false;
      console.log("Stoppping experiment with id " + this.experiment.id)
      return this.AutoDiscServerService.stopExperiment(this.experiment.id);
    }
    else{
      return new Observable<any>();
    }
  }

  downloadExperimentConfig(){
    if(this.experiment){
      let experimentConfig = {
        "experiment" : {
          "name" : this.experiment.name,
          "config" : this.experiment.config,
        },
        "system" : this.experiment.systems[0],
        "explorer": this.experiment.explorers[0],
        "input_wrappers": this.experiment.input_wrappers,
        "output_representations":this.experiment.output_representations,
        "callbacks":[],
        "logger_handlers":[]
      }
      var sJson = JSON.stringify(experimentConfig);
      var element = document.createElement('a');
      element.setAttribute('href', "data:text/json;charset=UTF-8," + encodeURIComponent(sJson));
      element.setAttribute('download', experimentConfig.experiment.name+".json");
      element.style.display = 'none';
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }
  }

  ngOnDestroy(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;    
  }
}
