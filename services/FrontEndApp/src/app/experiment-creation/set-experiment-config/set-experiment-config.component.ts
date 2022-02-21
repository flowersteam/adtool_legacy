import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-set-experiment-config',
  templateUrl: './set-experiment-config.component.html',
  styleUrls: ['./set-experiment-config.component.scss']
})
export class SetExperimentConfigComponent implements OnInit {
  
  objectKeys = Object.keys;
  
  @Input() currentConfig?: any;
  @Input() hosts?: any;

  constructor() { }

  ngOnInit(): void {
  }

}
