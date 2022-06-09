import { Component, OnInit, Input } from '@angular/core';
import { CreateNewExperimentService } from '../../services/create-new-experiment.service';

@Component({
  selector: 'app-set-module-config',
  templateUrl: './set-module-config.component.html',
  styleUrls: ['./set-module-config.component.scss']
})
export class SetModuleConfigComponent implements OnInit {

  objectKeys = Object.keys;
  
  @Input() currentModule?: any;
  @Input() module?: any;
  @Input() moduleItDependsOn?: any;
  @Input() displayInputOutputSpace? : Boolean;

  inputList : string[] = [];
  outputList : string[] = [];

  constructor(public createNewExperimentService: CreateNewExperimentService) { }

  ngOnInit(): void {
  }

  getInputList(){
    this.inputList = this.createNewExperimentService.makeDisplayableInputSpace(this.module, this.currentModule);
    return this.inputList;
  }

  getOtputList(){
    this.outputList = this.createNewExperimentService.makeDisplayableOutputSpace(this.module, this.currentModule);
    return this.outputList;
  }

  getInputOutput(){
    return this.createNewExperimentService.makeColorToDisplayInputOutputSpace(this.module, this.currentModule, this.inputList, this.outputList)
  }

  
}
