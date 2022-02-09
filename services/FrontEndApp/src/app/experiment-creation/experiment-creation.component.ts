import { Component, OnInit } from '@angular/core';


import { AutoDiscServerService } from '../services/auto-disc.service';
import { ExplorerSettings } from '../entities/explorer_settings';
import { SystemSettings } from '../entities/system_settings';
import { InputWrapperSettings } from '../entities/input_wrapper_settings';
import { OutputRepresentationSettings } from '../entities/output_representation_settings';
import { ExperimentSettings } from '../entities/experiment_settings';
import { Callback } from '../entities/callback';
import { Router } from '@angular/router';

import { JupyterService } from '../services/jupyter.service';
import { SetModuleComponent } from './set-module/set-module.component';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.scss']
})

export class ExperimentCreationComponent implements OnInit {

  objectKeys = Object.keys;

  //all modules get with api
  explorers: ExplorerSettings[] = [];
  systems: SystemSettings[] = [];
  input_wrappers: InputWrapperSettings[] = [];
  output_representations: OutputRepresentationSettings[] = [];
  callbacks: Callback[] = [];
  hosts: string[] = [];

  discovery_saving_keys_used: { [name: string]: boolean } = {};
  
  newExperiment = <ExperimentSettings>{};
  experiment_id:any = {};

  path_template_folder = "Templates";

  constructor(private AutoDiscServerService: AutoDiscServerService, private router: Router, private JupyterService: JupyterService, public setModuleComponent: SetModuleComponent) { }

  ngOnInit(): void {
    this.initExp();
    this.getExplorers();
    this.getInputWrappers();
    this.getSystems();
    this.getOutputRepresentations();
    this.getCallbacks();
    this.getHosts();
    
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

  getCallbacks(): void {
    this.AutoDiscServerService.getCallbacks()
    .subscribe(callbacks => this.callbacks = callbacks);
  }

  getHosts(): void {
    this.AutoDiscServerService.getHosts()
    .subscribe(hosts => this.hosts = hosts);
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
        host: "CHOOSE AN HOST",
      }
    }

    this.newExperiment.experiment.config.discovery_saving_keys = [];
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

// ##################   create exp ####################    
  createExp(){
    (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = true;
    var response = this.AutoDiscServerService.createExp(this.newExperiment).subscribe(res => {
      if(res == undefined){
        (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = false;
      }
      else{
        this.experiment_id = res["ID"]; 
        this.JupyterService.createNotebookDir(this.newExperiment.experiment.name, this.experiment_id, this.path_template_folder).subscribe(res => {this.router.navigate(["/experiment/"+this.experiment_id.toString()]);})
      }
    });    
  }

}
