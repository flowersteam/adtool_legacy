import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-experiment-config-details',
  templateUrl: './experiment-config-details.component.html',
  styleUrls: ['./experiment-config-details.component.scss']
})
export class ExperimentConfigDetailsComponent implements OnInit {

  @Input() experiment?: any;

  objectKeys = Object.keys;
  
  constructor() { }

  ngOnInit(): void {
  }

}
