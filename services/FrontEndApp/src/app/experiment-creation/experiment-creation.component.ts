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
// import { NOTINITIALIZED } from 'dns';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.css']
})

export class ExperimentCreationComponent implements OnInit {
  
  // animalControl = new FormControl('', Validators.required);
  // selectFormControl = new FormControl('', Validators.required);

  explorers: ExplorerSettings[] = [];
  systems: SystemSettings[] = [];
  input_wrappers: InputWrapperSettings[] = [];
  output_representations: OutputRepresentationSettings[] = [];
  discovery_saving_keys: string[] = []
  callbacks: Callback[] = [];

  // isActive = false;
  exp_name: string = "";
  nb_iteration: number = 1;
  nb_seed: number = 1;
  save_frequency: number = 1;
  discovery_saving_keys_used: { [name: string]: boolean } = {};
  dsku_list: string[] = []
  systemUsed: string = "CHOOSE A SYSTEM";
  explorerUsed: string = "CHOOSE AN EXPLORER";
  inputWrapperUsed: string = "CHOOSE AN INPUT WRAPPER";
  outputRepresentationUsed: string = "CHOOSE AN OUTPUT REPRESENTATION";
  
  newExperiment = <ExperimentSettings>{};

  actual_config_elt: any = {};

  // temp
  choice: any = {};
  constructor(private AutoDiscServerService: AutoDiscServerService) { }

  ngOnInit(): void {
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
    .subscribe(systems => this.systems = systems);
  }

  getInputWrappers(): void {
    this.AutoDiscServerService.getInputWrappers()
    .subscribe(input_wrappers => this.input_wrappers = input_wrappers);
  }
  getOutputRepresentations(): void {
    this.AutoDiscServerService.getOutputRepresentations()
    .subscribe(output_representations => this.output_representations = output_representations);
  }

  getDiscoverySavingKeys(): void {
    this.AutoDiscServerService.getDiscoverySavingKeys()
    .subscribe(discovery_saving_keys => {this.discovery_saving_keys = discovery_saving_keys.map(discovery_saving_key => discovery_saving_key.toString())
      this.setDiscoverySavingKeysUsed()});
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

  onCheckboxChange(key: string) {
    this.discovery_saving_keys_used[key] = !this.discovery_saving_keys_used[key]
  }

  setSystemUsed(system: string){
    this.systemUsed = system;
    console.log(this.systemUsed)
    console.log(this.systems)
  }

  setExplorerUsed(explorer: string){
    this.explorerUsed = explorer;
    console.log(this.explorerUsed)
    console.log(this.explorers)
  }

  setInputWrapperUsed(inputWrapper: string){
    this.inputWrapperUsed = inputWrapper;
    console.log(this.inputWrapperUsed)
    console.log(this.input_wrappers)
  }

  setOutputRepresentationUsed(outputRepresentation: string){
    this.outputRepresentationUsed = outputRepresentation;
    console.log(this.outputRepresentationUsed)
    console.log(this.output_representations)
    this.createExp();
  }

  getDiscoverySavingKeysUsed(){
    this.dsku_list = []
    for (let key in this.discovery_saving_keys_used) {
      if (this.discovery_saving_keys_used[key]){
        this.dsku_list.push(key)
      }
    }
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


// ##################   create exp ####################    
  createExp(){
    this.getDiscoverySavingKeysUsed()
    this.newExperiment.experiment = {
      "name": this.exp_name,
      config: {
        nb_iterations:this.nb_iteration,
        nb_seeds: this.nb_seed,
        save_frequency: this.save_frequency,
        discovery_saving_keys: this.dsku_list
      }
    }
    this.newExperiment.system = {
      "name": this.systemUsed,
      config: {}
    }
    this.newExperiment.explorer = {
      "name": this.explorerUsed,
      config: {}
    }
    this.newExperiment.input_wrappers = []
    this.newExperiment.input_wrappers.push({
      "name": this.inputWrapperUsed,
      config: {
        "wrapped_output_space_key": "init_state"
      }
    })
    this.newExperiment.output_representations = []
    this.newExperiment.output_representations.push({
      "name": this.outputRepresentationUsed,
      config: {}
    })
    this.newExperiment.callbacks = []
    // let my_exp = JSON.stringify(this.newExperiment)
    // console.log(my_exp)
    this.AutoDiscServerService.createExp(this.newExperiment).subscribe();
  }

}
