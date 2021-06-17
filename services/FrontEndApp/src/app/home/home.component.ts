import { Component, OnInit } from '@angular/core';

import { AppDbService } from '../services/app-db.service';
import { LightExperiment } from '../entities/light_experiment';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit {

  experiments: LightExperiment[] = [];
  
  constructor(private appDBService: AppDbService) { }

  ngOnInit() {
    this.getExperiments();
  }

  getExperiments(): void {
    this.appDBService.getLightExperiments()
    .subscribe(experiments => this.experiments = experiments);
  }

}
