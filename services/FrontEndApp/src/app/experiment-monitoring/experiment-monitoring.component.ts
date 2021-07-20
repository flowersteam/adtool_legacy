import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';

import { AppDbService } from '../services/app-db.service';
import { AutoDiscServerService } from '../services/auto-disc.service';
import { Experiment } from '../entities/experiment';
import { Observable, interval, Subscription } from 'rxjs';

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
  
  constructor(private appDBService: AppDbService, private AutoDiscServerService: AutoDiscServerService, private route: ActivatedRoute) { }

  ngOnInit() {
    this.getExperiment();
    this.resetAutoRefresh();
  }

  resetAutoRefresh(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
    this.intervalToSubscribe = interval(this.autoRefreshSeconds*1000);
    this.updateSubscription = this.intervalToSubscribe.subscribe(
      (val) => { this.getExperiment()});
  }

  getExperiment(): void {
    this.appDBService.getExperimentById(
      Number(this.route.snapshot.paramMap.get('id'))
    )
    .subscribe(experiment => {
      this.experiment = experiment;
      this.progressPercent = this.experiment.progress/experiment.config.nb_iterations*100;
      if (experiment.exp_status == 1){
        this.ellapsed = ((new Date().getTime() - new Date(experiment.created_on).getTime()) / 1000 / 60 / 60).toFixed(2);
      }     
    });
  }

  stopExperiment(): void {
    if (this.experiment != undefined){
      console.log("Stoppping experiment with id " + this.experiment.id)
      this.AutoDiscServerService.stopExperiment(this.experiment.id).subscribe(
        (val) => {this.getExperiment();}
      )
    }
  }

  ngOnDestroy(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
  }
}
