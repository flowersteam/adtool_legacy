import { Component, OnInit } from '@angular/core';


import { AutoDiscServerService } from '../services/auto-disc.service';
import { ExplorerSettings } from '../entities/explorer_settings';
import { SystemSettings } from '../entities/system_settings';
import { InputWrapperSettings } from '../entities/input_wrapper_settings';
import { OutputRepresentationSettings } from '../entities/output_representation_settings';
import { ExperimentSettings } from '../entities/experiment_settings';
import { Callback } from '../entities/callback';
import { NONE_TYPE } from '@angular/compiler';
import { FormControl, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { CdkDragDrop, DragDropModule, moveItemInArray }from '@angular/cdk/drag-drop';

import { JupyterService } from '../services/jupyter.service';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.scss']
})

export class ExperimentCreationComponent implements OnInit {
  
  // selectFormControl = new FormControl('', Validators.required);
  objectKeys = Object.keys;

  //all modules get with api
  explorers: ExplorerSettings[] = [];
  systems: SystemSettings[] = [];
  input_wrappers: InputWrapperSettings[] = [];
  output_representations: OutputRepresentationSettings[] = [];
  discovery_saving_keys: string[] = []
  callbacks: Callback[] = [];

  discovery_saving_keys_used: { [name: string]: boolean } = {};
  
  newExperiment = <ExperimentSettings>{};
  experiment_id:any = {};

  actual_config_elt: any = {};

  path_template_folder = "Templates";

  constructor(private AutoDiscServerService: AutoDiscServerService, private router: Router, private JupyterService: JupyterService) { }

  ngOnInit(): void {
    this.initExp();
    this.getExplorers();
    this.getInputWrappers();
    this.getSystems();
    this.getOutputRepresentations();
    this.getDiscoverySavingKeys();
    this.getCallbacks();
    
  }


  // ##################   get api data ####################
  getExplorers(): void {
    this.AutoDiscServerService.getExplorers()
    .subscribe(explorers => this.explorers = explorers);
  }

  getSystems(): void {
    this.AutoDiscServerService.getSystems()
    .subscribe(systems => {this.systems = systems;
                console.log(systems)});
  }

  getInputWrappers(): void {
    this.AutoDiscServerService.getInputWrappers()
    .subscribe(input_wrappers => this.input_wrappers = input_wrappers);
  }
  getOutputRepresentations(): void {
    this.AutoDiscServerService.getOutputRepresentations()
    .subscribe(output_representations => this.output_representations = output_representations);
  }

  // init exp
  getDiscoverySavingKeys(): void {
    this.AutoDiscServerService.getDiscoverySavingKeys()
    .subscribe(discovery_saving_keys => {this.discovery_saving_keys = discovery_saving_keys.map(discovery_saving_key => discovery_saving_key.toString())
      this.setDiscoverySavingKeysUsed(),
      this.getDiscoverySavingKeysUsed()});
  }

  getCallbacks(): void {
    this.AutoDiscServerService.getCallbacks()
    .subscribe(callbacks => this.callbacks = callbacks);
  }


  setDiscoverySavingKeysUsed():void{
  for (let key of this.discovery_saving_keys) {
      this.discovery_saving_keys_used[key.toString()] = true;   
    }
  }

  getDiscoverySavingKeysUsed(){
    this.newExperiment.experiment.config.discovery_saving_keys = []
    for (let key in this.discovery_saving_keys_used) {
      if (this.discovery_saving_keys_used[key]){
        this.newExperiment.experiment.config.discovery_saving_keys.push(key)
      }
    }
  }

  onCheckboxChange(key: string) {
    this.discovery_saving_keys_used[key] = !this.discovery_saving_keys_used[key]
    this.getDiscoverySavingKeysUsed()
  }

  

  setSystemUsed(system: string){
    this.newExperiment.system.name = system;
    this.newExperiment.system.config = {}
    let temp_system_config = this.getModuleByName(this.systems, system).config;
    for (let item in temp_system_config){
      this.newExperiment.system.config[item] = temp_system_config[item].default
    }
  }

  setExplorerUsed(explorer: string){
    this.newExperiment.explorer.name = explorer;
    this.newExperiment.explorer.config = {}
    let temp_explorer_config = this.getModuleByName(this.explorers, explorer).config;
    for (let item in temp_explorer_config){
      this.newExperiment.explorer.config[item] = temp_explorer_config[item].default
    }
  }

  addNewInputWrapperToUsed(inputWrapper: string){
    this.newExperiment.input_wrappers.push({
      name: inputWrapper,
      config: {wrapped_output_space_key:"empty"}
    })
    let temp_input_wrappers_config = this.getModuleByName(this.input_wrappers, inputWrapper).config;
    for (let item in temp_input_wrappers_config){
      this.newExperiment.input_wrappers[this.newExperiment.input_wrappers.length - 1].config[item] = temp_input_wrappers_config[item].default
    }
  }

  removeInputwrapperToUsed(index: number){
    this.newExperiment.input_wrappers.splice(index, 1);
  }

  addNewOutputRepresentationToUsed(outputRepresentation: string){
    this.newExperiment.output_representations.push({
      name: outputRepresentation,
      config: {wrapped_input_space_key:"empty"}
    })
    let temp_output_representation_config = this.getModuleByName(this.output_representations, outputRepresentation).config;
    for (let item in temp_output_representation_config){
      this.newExperiment.output_representations[this.newExperiment.output_representations.length - 1].config[item] = temp_output_representation_config[item].default
    }
  }

  removeOutputRepresentationToUsed(index: number){
    this.newExperiment.output_representations.splice(index, 1);
  }


  // ############# to make config definition design in html ########## 
  getModuleByName(modules: any, name: string):any{
    for (let module of modules) {
      if(module.name == name)
        return(module)      
    }
  }
  
  setActualConfigElt(elt: any): any{
    this.actual_config_elt = elt;
  }

  calculStep(){
    return(Math.trunc((this.actual_config_elt.max - this.actual_config_elt.min)/20))
  }

  
  // ################ init exp #################
  initExp(){
    // general config part
    this.newExperiment.experiment = {
      name: "unnamed-exp",
      config: {
        nb_iterations:1,
        nb_seeds:1,
        save_frequency:1,
      }
    }
    if(this.discovery_saving_keys_used != undefined){
      this.getDiscoverySavingKeysUsed()
    }

    //system part
    this.newExperiment.system = {
      name: "CHOOSE A SYSTEM",
      config: {}
    }

    this.newExperiment.explorer = {
      name: "CHOOSE AN EXPLORER",
      config: {}
    }

    this.newExperiment.input_wrappers = []
    this.newExperiment.output_representations = []
    this.newExperiment.callbacks = []
    this.newExperiment.logger_handlers = []

  }

  drop(event: CdkDragDrop<string[]>, my_dragdrop_array: any[]) {
    moveItemInArray(my_dragdrop_array, event.previousIndex, event.currentIndex);
  }

  consolelog(index: any){
    console.log(index)
    // for(let i of index){
    //   console.log(i)
    // }
  }

// ##################   create exp ####################    
  createExp(){
    this.AutoDiscServerService.createExp(this.newExperiment).subscribe(res => {
      this.experiment_id = res["ID"]; 
      this.JupyterService.createNotebookDir(this.newExperiment.experiment.name, this.experiment_id, this.path_template_folder).subscribe(res => {this.router.navigate(["/experiment/"+this.experiment_id.toString()]);})
      
    });
  }

}
