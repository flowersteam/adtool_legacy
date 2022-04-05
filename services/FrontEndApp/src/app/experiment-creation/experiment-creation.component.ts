import { Component, OnInit } from '@angular/core';

import { AutoDiscServerService } from '../services/REST-services/auto-disc.service';
import { Router } from '@angular/router';

import { JupyterService } from '../services/jupyter.service';
import { CreateNewExperimentService } from '../services/create-new-experiment.service';
import { ToasterService } from '../services/toaster.service';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.scss']
})

export class ExperimentCreationComponent implements OnInit {

  objectKeys = Object.keys;  

  constructor(public createNewExperimentService: CreateNewExperimentService, private AutoDiscServerService: AutoDiscServerService, 
              private router: Router, private JupyterService: JupyterService, private toasterService: ToasterService) { }

  ngOnInit(): void {
    this.createNewExperimentService.setAllConfigs();
    this.createNewExperimentService.initExperiment();
  }
  
  createExperiment(){
    if(this.createNewExperimentService.checkNewExperimentSet()){
      this.toasterService.showInfo("Experiment Launch", "experiment start");
      let experiment_id:any = {};
      let path_template_folder = "Templates";
      (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = true;
      this.AutoDiscServerService.createExperiment(this.createNewExperimentService.newExperiment).subscribe(res => {
        if(!res.success){
          (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = false;
          this.toasterService.showError(res.message ?? '', "Error creating experiment", {timeOut: 0, extendedTimeOut: 0});
        }
        else{
          this.toasterService.showSuccess("Experiment start", "Experiment Run");
          experiment_id = res.data["ID"];
          if(this.createNewExperimentService.newExperiment.experiment.name){
            this.JupyterService.createNotebookDir(this.createNewExperimentService.newExperiment.experiment.name, experiment_id, path_template_folder).subscribe(res => {this.router.navigate(["/experiment/"+experiment_id.toString()]);})
          }        
        }
      });
    }
  }

}
