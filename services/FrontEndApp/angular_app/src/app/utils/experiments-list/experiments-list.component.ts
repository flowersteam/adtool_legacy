import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-experiments-list',
  templateUrl: './experiments-list.component.html',
  styleUrls: ['./experiments-list.component.scss']
})
export class ExperimentsListComponent implements OnInit {

  constructor() { }

  @Input() experiments?: any;
  @Input() needRouter?: boolean;
  @Input() needChoice?: boolean;
  @Input() experiment?: any;
  @Output() experimentChange = new EventEmitter();
  @Output() triggerParentMethod = new EventEmitter<any>();
  @Input() searchText: string = "";

  ngOnInit(): void {
  }

  callParent(experiment : any){
    if(this.triggerParentMethod.observers.length >= 1){
      this.experiment = experiment;
      this.experimentChange.emit(this.experiment);
      this.triggerParentMethod.next();
    }
  }
}