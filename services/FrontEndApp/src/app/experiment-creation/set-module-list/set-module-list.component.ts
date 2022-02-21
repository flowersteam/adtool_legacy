import { Component, OnInit, Input } from '@angular/core';
import { CdkDragDrop, DragDropModule, moveItemInArray }from '@angular/cdk/drag-drop';
import { cloneDeep } from 'lodash'

import { CreateNewExperimentService } from '../../services/create-new-experiment.service';


@Component({
  selector: 'app-set-module-list',
  templateUrl: './set-module-list.component.html',
  styleUrls: ['./set-module-list.component.scss']
})
export class SetModuleListComponent implements OnInit {

  objectKeys = Object.keys;

  @Input() currentModuleList?: any; // return by reference
  @Input() modules?: any;
  @Input() currentSystem? : any;
  @Input() systems : any;
  @Input() displayInputOutputSpace? : Boolean;
  @Input() key? : string;
  @Input() spaceItDependsOn? : any;

  @Input() customModules : any;

  constructor(public createNewExperimentService: CreateNewExperimentService) { }

  ngOnInit(): void {
  }

  drop(event: CdkDragDrop<string[]>, dragdrop_array: any[]) {
    moveItemInArray(dragdrop_array, event.previousIndex, event.currentIndex);
    moveItemInArray(this.customModules, event.previousIndex, event.currentIndex); //move custom config module link to a module
    let response = this.createNewExperimentService.defineCustomModuleList(this.currentModuleList, this.customModules, this.modules, this.key, this.spaceItDependsOn);
    this.customModules = response[0]
    this.currentModuleList = response[1]
  }

  remove(index : number){
    let response = this.createNewExperimentService.removeModuleToUse(this.currentModuleList, this.customModules, this.modules, this.key, this.spaceItDependsOn, index);
    this.customModules = response[0]
    this.currentModuleList = response[1]
  }

}
