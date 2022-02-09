import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-set-experiment-config',
  templateUrl: './set-experiment-config.component.html',
  styleUrls: ['./set-experiment-config.component.scss']
})
export class SetExperimentConfigComponent implements OnInit {
  
  objectKeys = Object.keys;
  
  @Input() currentConfig?: any; // return by reference
  @Input() hosts?: any; // return by reference

  constructor() { }

  ngOnInit(): void {
  }

  setHostUsed(host: string){
    this.currentConfig.config.host = host;
  }

}
