import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-set-module',
  templateUrl: './set-module.component.html',
  styleUrls: ['./set-module.component.scss']
})
export class SetModuleComponent implements OnInit {

  objectKeys = Object.keys;

  @Input() currentModule?: any; // return by reference

  @Input() modules?: any;
  @Input() moduleItDependsOn?: any;

  @Input() displayInputOutputSpace? : Boolean;

  constructor() { }

  ngOnInit(): void {
  }

  static getModuleByName(modules: any, name: string):any{
    for (let module of modules) {
      if(module.name == name)
        return(module)      
    }
  }

  getModuleByName = SetModuleComponent.getModuleByName

  setModuleUsed(module: string){
    if(this.currentModule.config.wrapped_output_space_key != undefined){
      this.currentModule.config = {wrapped_output_space_key:"empty"}
    }
    else if(this.currentModule.config.wrapped_input_space_key != undefined){
      this.currentModule.config = {wrapped_input_space_key:"empty"}
    }
    else{
      this.currentModule.config = {}
    }
    this.currentModule.name = module; 
    let temp_module_config = this.getModuleByName(this.modules, module).config;
    for (let item in temp_module_config){
      this.currentModule.config[item] = temp_module_config[item].default
    }
  }

}
