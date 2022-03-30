import { Component, Inject, OnInit } from '@angular/core';
import {MAT_DIALOG_DATA, MatDialogRef} from "@angular/material/dialog";
import { AppDbService } from '../../../services/app-db.service';
import { Experiment } from '../../../entities/experiment';

@Component({
  selector: 'app-choose-previous-experiment',
  templateUrl: './choose-previous-experiment.component.html',
  styleUrls: ['./choose-previous-experiment.component.scss']
})
export class ChoosePreviousExperimentComponent implements OnInit {
  
  constructor(private appDBService: AppDbService, 
              private dialogRef: MatDialogRef<ChoosePreviousExperimentComponent>) { }
  
  experiments: Experiment[] = [];
  sortByDateAsc: boolean = true; 
  searchText = '';

  
  ngOnInit() {
    this.getExperiments();
  }

  getExperiments(): void {
    this.appDBService.getLightExperiments()
    .subscribe(experiments => {
      this.sortByDateAsc = true; 
      this.experiments = experiments;
      this.sortExperimentsByDate();
    });
  }

  sortExperimentsByDate(): void {
    this.sortByDateAsc = !this.sortByDateAsc;
    this.experiments.sort((a, b) => {
      return this.sortByDateAsc ? +new Date(a.created_on) - +new Date(b.created_on) : +new Date(b.created_on) - +new Date(a.created_on);
    });
  }

  selectPreviousExperiment(experiment : any){
    this.dialogRef.close(experiment.id);
  }

}
