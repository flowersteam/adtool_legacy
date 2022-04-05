import { Injectable } from '@angular/core';
import { cloneDeep } from 'lodash'

import { AutoDiscServerService } from '../services/REST-services/auto-disc.service';
import { ExperimentSettings } from '../entities/experiment_settings';
import { ToasterService } from '../services/toaster.service';

@Injectable({
  providedIn: 'root'
})
export class CreateNewExperimentService {

  allConfig : any = []; // all modules config give by api
  customConfig: any = []; /* some cutsom module config use for input_wrapper and output_representation. 
                          We need it because config of input_wrapper an output_representation depends on current context*/
  currentSytemName : string | undefined;
  newExperiment = <ExperimentSettings>{};
  constructor(private AutoDiscServerService: AutoDiscServerService, private toasterService: ToasterService) { }

  

  // ##################   get api data ####################
  setAllConfigs(){
    this.getExplorers();
    this.getInputWrappers();
    this.getSystems();
    this.getOutputRepresentations();
    this.getCallbacks();
    this.getHosts();
    this.customConfig.input_wrappers = [];
    this.customConfig.output_representations = [];
  }

  getExplorers(): void {
    this.AutoDiscServerService.getExplorers().subscribe(response => {
      if(response.success){
        this.allConfig.explorers = response.data;
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting explorers", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  getSystems(): void {
    this.AutoDiscServerService.getSystems().subscribe(response => {
      if(response.success){
        this.allConfig.systems = response.data;
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting systems", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  getInputWrappers(): void {
    this.AutoDiscServerService.getInputWrappers().subscribe(response => {
      if(response.success){
        this.allConfig.input_wrappers = response.data;
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting input wrappers", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }
  getOutputRepresentations(): void {
    this.AutoDiscServerService.getOutputRepresentations().subscribe(response => {
      if(response.success){
        this.allConfig.output_representations = response.data;
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting output representations", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  getCallbacks(): void {
    this.AutoDiscServerService.getCallbacks().subscribe(response => {
      if(response.success){
        this.allConfig.callbacks = response.data;
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting callbacks", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  getHosts(): void {
    this.AutoDiscServerService.getHosts().subscribe(response => {
      if(response.success){
        this.allConfig.hosts = response.data;
        this.newExperiment.experiment.config.host = this.allConfig.hosts[0];
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting hosts", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  // ##################   init new experiment ####################
  initExperiment(){
    // general config part
    this.newExperiment.experiment = {
      name: undefined,
      config: {
        nb_iterations:1,
        nb_seeds:1,
        save_frequency:1,
        host: undefined,
      }
    }

    this.newExperiment.experiment.config.discovery_saving_keys = [];
    //system part
    this.newExperiment.system = {
      name: undefined,
      config: {}
    }

    this.newExperiment.explorer = {
      name: undefined,
      config: {}
    }

    this.newExperiment.input_wrappers = []
    this.newExperiment.output_representations = []
    this.newExperiment.callbacks = []
    this.newExperiment.logger_handlers = []

    this.currentSytemName = this.newExperiment.system.name;
  }

  // ##################       utils           ####################

  checkNewExperimentSet(){
    let experimentIsOk = true;
    if(this.newExperiment.input_wrappers.length == 0){
      this.toasterService.showInfo("No Input Wrappers defined", "Input Wrappers");
    }
    if(this.newExperiment.output_representations.length == 0){
      this.toasterService.showInfo("No output representations defined", "Output Representations");
    }
    if(this.newExperiment.experiment.name == undefined || this.newExperiment.experiment.name == ""){
      this.toasterService.showWarning("Experiment has no name", "General Information");
      experimentIsOk = false;
    }
    if(this.newExperiment.system.name == undefined){
      this.toasterService.showWarning("No system defined", "System");
      experimentIsOk = false;
    }
    if(this.newExperiment.explorer.name == undefined){
      this.toasterService.showWarning("No explorer defined", "Explorer");
      experimentIsOk = false;
    }
    return experimentIsOk;
  }

  checkExperimentName(){
    let alphanumericRegex = /^[a-zA-Z0-9_-]+$/g;
    if(this.newExperiment.experiment.name && !alphanumericRegex.test(this.newExperiment.experiment.name)){
      let spaceRegex = /(\s)/g;
      let notAlphanumericRegex = /[^a-zA-Z0-9_-]+/g
      this.newExperiment.experiment.name = this.newExperiment.experiment.name?.trim();
      this.newExperiment.experiment.name =this.newExperiment.experiment.name?.replace(spaceRegex, "_");
      this.newExperiment.experiment.name =this.newExperiment.experiment.name?.replace(notAlphanumericRegex, "");
      this.newExperiment.experiment.name = this.newExperiment.experiment.name?.replace(/^_+|_+$/g, ''); // equal to trim("_") in other language
      this.toasterService.showWarning("the name of the experiment has been changed to match alphanumeric characters", "Experiment name");
    }
  }

  setModuleUse(currentModule : any, moduleName: string, modules : any){
    currentModule.config = {};
    currentModule.name = moduleName; 
    let temp_module_config = CreateNewExperimentService.getModuleByName(modules, moduleName).config;
    for (let item in temp_module_config){
      currentModule.config[item] = temp_module_config[item].default
    }

    this.moduleChange()
  }

  static getModuleByName(modules: any, name: string | undefined):any{
    if(modules != undefined){
      for (let module of modules) {
        if(module.name == name)
          return(module)      
      }
    } 
  }
  getModuleByName = CreateNewExperimentService.getModuleByName;

  moduleChange(){
    if(this.newExperiment.system.name != this.currentSytemName){
      this.newExperiment.input_wrappers = [];
      this.newExperiment.output_representations = [];
      this.customConfig.input_wrappers = [];
      this.customConfig.output_representations = [];
      this.currentSytemName = this.newExperiment.system.name;
    }
  }

  makeDisplayableInputSpace(module : any, currentModule: any){
    let input_space;
    if(module.config.wrapped_output_space_key != undefined){
      input_space = Object.keys(module.output_space);
      let index = input_space.indexOf(currentModule.config.wrapped_output_space_key);
      delete input_space[index];
      input_space = input_space.concat(Object.keys(module.input_space));
    }
    else{
      input_space = Object.keys(module.input_space);
    }
    return(input_space)
  }

  makeDisplayableOutputSpace(module : any, currentModule: any){
    let output_space;
    if(module.config.wrapped_input_space_key != undefined){
      output_space = Object.keys(module.output_space);
      output_space = output_space.concat(Object.keys(module.input_space));
      let index = output_space.indexOf(currentModule.config.wrapped_input_space_key);
      delete output_space[index];
    }
    else{
      output_space = Object.keys(module.output_space);
    }
    return(output_space)
  }

  makeColorToDisplayInputOutputSpace(module : any, currentModule: any, inputList : string[], outputList : string[]){
    let inputColor : string[] = [];
    let outputColor : string[] = [];
    let inputSpace = Object.keys(module.input_space);
    let outputSpace = Object.keys(module.output_space);
    if(module.config.wrapped_output_space_key != undefined){
      for(let input of inputList){
        if(inputSpace.indexOf(input) >= 0){
          inputColor.push(input);
        }
      }
      outputColor.push(currentModule.config.wrapped_output_space_key);
    }
    else if(module.config.wrapped_input_space_key != undefined){
      for(let output of outputList){
        if(outputSpace.indexOf(output) >= 0){
          outputColor.push(output);
        }
      }
      inputColor.push(currentModule.config.wrapped_input_space_key);
    }
    return [inputColor, outputColor];
  }

  // ##################   custom config ####################

    setAllCustomModulesFromUseModule(){
      this.customConfig.input_wrappers = [];
      this.customConfig.output_representations = [];
      this.setOneTypeOfCustomModulesFromUseModule(this.customConfig.input_wrappers, this.allConfig.input_wrappers, this.newExperiment.input_wrappers, "input_wrapper");
      this.setOneTypeOfCustomModulesFromUseModule(this.customConfig.output_representations, this.allConfig.output_representations, this.newExperiment.output_representations, "output_representation");
    }

    setOneTypeOfCustomModulesFromUseModule(customModules : any, modules:any, useModules: any, moduleType : string){
      let key : string;
      let spaceItDependsOn : any;
      if(moduleType == "input_wrapper"){
        key = "wrapped_output_space_key";
        spaceItDependsOn = this.getModuleByName(this.allConfig.systems, this.newExperiment.system.name).input_space;
        customModules = this.customConfig.input_wrappers;
      }
      else if(moduleType == "output_representation"){
        key = "wrapped_input_space_key";
        spaceItDependsOn = this.getModuleByName(this.allConfig.systems, this.newExperiment.system.name).output_space;
        customModules = this.customConfig.output_representations;
      }

      useModules.forEach((value:any, i:number) => {
        this.addCustomModuleToList(customModules, modules, useModules[i].name, key, spaceItDependsOn)
      });
    }

    addCustomModuleToList(customModules :any, modules :any, moduleName: string, key : string|undefined, spaceItDependsOn : any){
      let index : number;
      modules[0].input_space != undefined ? index = 0 : index =   customModules.length;
      customModules.splice(index, 0, cloneDeep(CreateNewExperimentService.getModuleByName(modules, moduleName)));
      
      if(customModules[index].input_space == undefined){
        if(customModules.length > 1){spaceItDependsOn = customModules[index-1].output_space}
        customModules[index].input_space = spaceItDependsOn;
      }
      else if(customModules[index].output_space == undefined){
        if(customModules.length > 1){spaceItDependsOn = customModules[index+1].input_space}
        customModules[index].output_space = spaceItDependsOn;
      }
      if(key){
        customModules[index].config[key] = { 
          default:Object.keys(spaceItDependsOn)[0], 
          possible_values: Object.keys(spaceItDependsOn),  
          type: 'STRING'
        }
      }
      return [index, customModules[index]];
    }

    defineCustomModuleList(currentModuleList : any, customModules: any, modules :any, key : string|undefined, spaceItDependsOn : any){
      let customModulesNames : string[] = [];
      for(let customModule of customModules){
        customModulesNames.push(customModule.name)
      }
      customModules = [];
      currentModuleList = []
      // currentModuleList = [];
      if(key == "wrapped_output_space_key"){
        customModulesNames = customModulesNames.reverse()
      }
      for(let moduleName of customModulesNames){
        this.addNewModuleToUse(currentModuleList, customModules, modules, key, spaceItDependsOn, moduleName)
      }
      return([customModules, currentModuleList])
    }
  
    setUseModuleFromCustomModule(customModule : any){
      let useModule :any;
      useModule = {
        name: customModule.name,
        config: {}
      }
      for (let item in customModule.config){
        useModule.config[item] = customModule.config[item].default
      }
      return(useModule)
    }

    addNewModuleToUse(currentModuleList : any, customModules: any, modules :any, key : string|undefined, spaceItDependsOn : any, moduleName: string){
      if(spaceItDependsOn == undefined){
        this.toasterService.showWarning("Choose a system first", "System");
      }
      else{
        let index : number;
        let newCustomModule : any;
        let response = this.addCustomModuleToList(customModules, modules ,moduleName, key, spaceItDependsOn);
        index = response[0];
        newCustomModule = response[1];
        currentModuleList.splice(index, 0, this.setUseModuleFromCustomModule(newCustomModule));
      }
    }

    removeModuleToUse(currentModuleList : any, customModules: any, modules :any, key : string|undefined, spaceItDependsOn : any, index: number){
      currentModuleList.splice(index, 1);
      customModules.splice(index, 1); //remove custom config module link to a module
      return this.defineCustomModuleList(currentModuleList, customModules, modules, key, spaceItDependsOn);
    }
}
