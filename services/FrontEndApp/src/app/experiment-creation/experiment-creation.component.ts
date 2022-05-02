import { Component, OnInit } from '@angular/core';
import {MatDialog} from '@angular/material/dialog';

import { AutoDiscServerService } from '../services/auto-disc.service';
import { Router } from '@angular/router';

import { JupyterService } from '../services/jupyter.service';
import { CreateNewExperimentService } from '../services/create-new-experiment.service';
import { ToasterService } from '../services/toaster.service';
import { PreparingLogComponent } from './preparing-log/preparing-log.component';
import { AppDbService } from '../services/app-db.service';

@Component({
  selector: 'app-experiment-creation',
  templateUrl: './experiment-creation.component.html',
  styleUrls: ['./experiment-creation.component.scss']
})

export class ExperimentCreationComponent implements OnInit {

  objectKeys = Object.keys;  

  constructor(public createNewExperimentService: CreateNewExperimentService, private AutoDiscServerService: AutoDiscServerService, 
              private router: Router, private JupyterService: JupyterService, private toasterService: ToasterService,
              public dialog: MatDialog, private appDBService: AppDbService) { }

  ngOnInit(): void {
    this.createNewExperimentService.setAllConfigs();
    this.createNewExperimentService.initExperiment();
  }

  openDialog(id : number): void {
    const dialogRef = this.dialog.open(PreparingLogComponent, {data:{experiment_id: id}, disableClose: true});
    this.router.events.subscribe(() => {dialogRef.close();});
    dialogRef.afterClosed().subscribe(result => {
      if(result != undefined){
        // this.setExperimentWithPreviousExperiment(result);
      }
    });
  }

  sleep(ms:number) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }
  
  createExperiment(){
    if(this.createNewExperimentService.checkNewExperimentSet()){
      this.toasterService.showInfo("Experiment Launch", "experiment start");
      let experiment_id:any = {};
      let path_template_folder = "Templates";
      let experiment_status = 4; // preparing status
      (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = true;
      var response = this.AutoDiscServerService.createExperiment(this.createNewExperimentService.newExperiment).subscribe(res => {
        if(res.status < 200 || res.status > 299 ){
          (<HTMLInputElement> document.getElementById("btn_create_exp")).disabled = false;
          this.toasterService.showError(res.error, "unexpected error occured", {timeOut: 0, extendedTimeOut: 0});
        }
        else{
          this.toasterService.showSuccess("Experiment start", "Experiment Run");
          experiment_id = res["ID"];
          this.openDialog(experiment_id);
          if(this.createNewExperimentService.newExperiment.experiment.name){
            this.JupyterService.createNotebookDir(this.createNewExperimentService.newExperiment.experiment.name, experiment_id, path_template_folder).subscribe(async res => {
              while(experiment_status == 4){
                this.appDBService.getExperimentById(experiment_id).subscribe((experiment: { exp_status: number; }) => {
                  experiment_status = experiment.exp_status;
                });
                await this.sleep(1000);
              }
              this.router.navigate(["/experiment/"+experiment_id.toString()]);
            })
          }        
        }
      });
    }
  }

}
