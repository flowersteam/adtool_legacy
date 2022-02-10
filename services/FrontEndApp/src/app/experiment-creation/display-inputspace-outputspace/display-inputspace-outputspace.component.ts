import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-display-inputspace-outputspace',
  templateUrl: './display-inputspace-outputspace.component.html',
  styleUrls: ['./display-inputspace-outputspace.component.scss']
})
export class DisplayInputspaceOutputspaceComponent implements OnInit {

  objectKeys = Object.keys;

  @Input() currentModule? : any;
  @Input() currentModuleSetting?: any; // return by reference
  @Input() displayInputOutputSpace?: any;
  @Input() moduleItDependsOn? : any;

  constructor() { }

  ngOnInit(): void {}

}
