import { Component, OnInit, Input } from '@angular/core';
import { CdkDragDrop, DragDropModule, moveItemInArray }from '@angular/cdk/drag-drop';
import { SetModuleComponent } from '../set-module/set-module.component';


@Component({
  selector: 'app-set-module-list',
  templateUrl: './set-module-list.component.html',
  styleUrls: ['./set-module-list.component.scss']
})
export class SetModuleListComponent implements OnInit {

  objectKeys = Object.keys;

  @Input() moduleType?: string; // return by reference
  @Input() currentModuleList?: any; // return by reference
  @Input() modules?: any;
  @Input() currentSystem? : any;
  @Input() systems : any;
  @Input() displayInputOutputSpace? : Boolean;

  constructor(public setModuleComponent: SetModuleComponent) { }

  ngOnInit(): void {
  }

  drop(event: CdkDragDrop<string[]>, my_dragdrop_array: any[]) {
    moveItemInArray(my_dragdrop_array, event.previousIndex, event.currentIndex);
  }

  removeModuleToUsed(index: number){
    this.currentModuleList.splice(index, 1);
  }

  addNewModuleToUsed(module: string){
    let config = {};
    if(this.moduleType == "OUTPUT REPRESENTATION"){
      config = {wrapped_input_space_key:"empty"};
    }
    else if(this.moduleType == "INPUT WRAPPER"){
      config = {wrapped_output_space_key:"empty"};
    }
    this.currentModuleList.push({
      name: module,
      config: config
    })
    let temp_input_wrappers_config = this.setModuleComponent.getModuleByName(this.modules, module).config;
    for (let item in temp_input_wrappers_config){
      this.currentModuleList[this.currentModuleList.length - 1].config[item] = temp_input_wrappers_config[item].default
    }
  }

}
