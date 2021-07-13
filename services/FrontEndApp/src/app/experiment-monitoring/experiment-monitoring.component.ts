import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';

import { AppDbService } from '../services/app-db.service';
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
  private intervalToSubscribe: Observable<number> | undefined;
  private updateSubscription: Subscription | undefined;
  public autoRefreshSeconds: number = 30;
  
  constructor(private appDBService: AppDbService, private route: ActivatedRoute) { }

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
      if (experiment.exp_status == 1){
        this.ellapsed = ((new Date().getTime() - new Date(experiment.created_on).getTime()) / 1000 / 60 / 60).toFixed(2);
      }     
    });
  }

  ngOnDestroy(): void{
    this.updateSubscription?.unsubscribe();
    this.intervalToSubscribe = undefined;
  }
}
