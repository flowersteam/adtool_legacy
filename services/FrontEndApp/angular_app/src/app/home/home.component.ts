import { Component, OnInit } from '@angular/core';

import { AppDbService } from '../services/REST-services/app-db.service';
import { ToasterService } from '../services/toaster.service';

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
  
  constructor(private appDBService: AppDbService, private toasterService: ToasterService) { }

  ngOnInit() {
    this.getExperiments();
  }

  getExperiments(): void {
    this.appDBService.getLightExperiments()
    .subscribe(response => {
      if(response.success){
        this.sortByDateAsc = true; 
        this.experiments = response.data ?? [];
        this.sortExperimentsByDate();
      }
      else {
        this.toasterService.showError(response.message ?? '', "Error listing experiments");
      }
    });
  }

  sortExperimentsByDate(): void {
    this.sortByDateAsc = !this.sortByDateAsc;
    this.experiments.sort((a, b) => {
      return this.sortByDateAsc ? +new Date(a.created_on) - +new Date(b.created_on) : +new Date(b.created_on) - +new Date(a.created_on);
    });
  }

}
