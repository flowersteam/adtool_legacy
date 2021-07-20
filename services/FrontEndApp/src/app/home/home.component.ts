import { Component, OnInit } from '@angular/core';

import { AppDbService } from '../services/app-db.service';
import { Experiment } from '../entities/experiment';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  experiments: Experiment[] = [];
  sortByDateAsc: boolean = true; 
  searchText = '';
  
  constructor(private appDBService: AppDbService) { }

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

}
