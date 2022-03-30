import { Component, OnInit, AfterViewInit, ViewChild, Directive } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/app-db.service';
import { AutoDiscServerService } from '../services/auto-disc.service';
import { Experiment } from '../entities/experiment';
import { Observable, interval, Subscription, empty } from 'rxjs';

import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-experiment-monitoring',
  templateUrl: './experiment-monitoring.component.html',
  styleUrls: ['./experiment-monitoring.component.scss'],
})
export class ExperimentMonitoringComponent implements OnInit {

  experiment: Experiment | undefined;
  public ellapsed: string | undefined;
  public progressPercent:string = "0";
  private intervalToSubscribe: Observable<number> | undefined;
  private updateSubscription: Subscription | undefined;

  public autoRefreshSeconds: number = 5;
  objectKeys = Object.keys;
  urlSafe: SafeResourceUrl | undefined;
  
  constructor(private appDBService: AppDbService, private AutoDiscServerService: AutoDiscServerService, private route: ActivatedRoute,
              public sanitizer: DomSanitizer) { }

  ngOnInit() {
    this.refreshExperiment();
    this.resetAutoRefresh();
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
