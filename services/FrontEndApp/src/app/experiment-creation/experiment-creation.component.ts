import { Component, OnInit } from '@angular/core';
import { AutoDiscServerService } from '../services/REST-services/auto-disc.service';
import { Router } from '@angular/router';
import { JupyterService } from '../services/jupyter.service';
import { CreateNewExperimentService } from '../services/create-new-experiment.service';
import { ToasterService } from '../services/toaster.service';
import { AppDbService } from '../services/REST-services/app-db.service';
import { PreparingLogService } from '../services/preparing-log.service';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.scss']
})

export class ExperimentCreationComponent implements OnInit {

  objectKeys = Object.keys;  

  constructor(public createNewExperimentService: CreateNewExperimentService, private AutoDiscServerService: AutoDiscServerService, 
              private router: Router, private JupyterService: JupyterService, private toasterService: ToasterService,
              private appDBService: AppDbService, private preparingLogService: PreparingLogService) { }

  ngOnInit(): void {
    this.createNewExperimentService.setAllConfigs();
    this.createNewExperimentService.initExperiment();
  }
  
  createExperiment(){
    if(this.createNewExperimentService.checkNewExperimentSet()){
      this.toasterService.showInfo("Experiment Launch", "experiment start");
      let experiment_id:any = {};
      let path_template_folder = "Templates";
      let experiment_status = 4; // preparing status
      (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = true;
      this.AutoDiscServerService.createExperiment(this.createNewExperimentService.newExperiment).subscribe(res => {
        if(!res.success){
          (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = false;
          this.toasterService.showError(res.message ?? '', "Error creating experiment", {timeOut: 0, extendedTimeOut: 0});
        }
        else{
          this.toasterService.showSuccess("Experiment start", "Experiment Run");
          experiment_id = res.data["ID"];
          this.preparingLogService.openDialog(experiment_id);
          if(this.createNewExperimentService.newExperiment.experiment.name){
            this.JupyterService.createNotebookDir(this.createNewExperimentService.newExperiment.experiment.name, experiment_id, path_template_folder).subscribe(async res => {
              while(experiment_status == 4){
                this.appDBService.getExperimentById(experiment_id).subscribe((experiment: any) => {
                  experiment_status = experiment.data.exp_status;
                });
                await this.preparingLogService.sleep(1000);
              }
              if(experiment_status != 4){
                this.router.navigate(["/experiment/"+experiment_id.toString()]);
              }
            })
          }        
        }
      });
    }
  }

}
