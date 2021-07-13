import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { Location } from '@angular/common';

import { AppDbService } from '../services/app-db.service';
import { Experiment } from '../entities/experiment';

@Component({
  selector: 'app-experiment-monitoring',
  templateUrl: './experiment-monitoring.component.html',
  styleUrls: ['./experiment-monitoring.component.scss']
})
export class ExperimentMonitoringComponent implements OnInit {

  experiment: Experiment | undefined;
  public ellapsed: Number | undefined;
  
  constructor(private appDBService: AppDbService, private route: ActivatedRoute) { }

  ngOnInit() {
    this.getExperiment();
  }

  getExperiment(): void {
    this.appDBService.getExperimentById(
      Number(this.route.snapshot.paramMap.get('id'))
    )
    .subscribe(experiment => {
      this.experiment = experiment;
      if (experiment.exp_status == 1){
        this.ellapsed = (new Date().getTime() - new Date(experiment.created_on).getTime()) / 1000 / 60;
      }     
    });
  }

}
