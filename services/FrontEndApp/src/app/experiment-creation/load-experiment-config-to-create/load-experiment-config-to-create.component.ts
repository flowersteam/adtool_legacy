import { HttpClient } from '@angular/common/http';
import { Component, OnInit, Input } from '@angular/core';
import {MatDialog} from '@angular/material/dialog';

import { CreateNewExperimentService } from '../../services/create-new-experiment.service';
import { ChoosePreviousExperimentComponent } from './choose-previous-experiment/choose-previous-experiment.component';
import { AppDbService } from '../../services/app-db.service';

@Component({
  selector: 'app-load-experiment-config-to-create',
  templateUrl: './load-experiment-config-to-create.component.html',
  styleUrls: ['./load-experiment-config-to-create.component.scss']
})
export class LoadExperimentConfigToCreateComponent implements OnInit {

  @Input() currentExperiment?: any;

  previous_experiment_id = undefined;
  fileName = '';
  fileContent : string | ArrayBuffer = '';
  constructor(private http: HttpClient, public createNewExperimentService: CreateNewExperimentService, private appDBService: AppDbService, public dialog: MatDialog) { }
  
  ngOnInit(): void {}

  openDialog(): void {
    const dialogRef = this.dialog.open(ChoosePreviousExperimentComponent, {data:{id: this.previous_experiment_id}});
    dialogRef.afterClosed().subscribe(result => {
      if(result != undefined){
        this.setExperimentWithPreviousExperiment(result);
      }
    });
  }

  downloadJson(){
    var sJson = JSON.stringify(this.currentExperiment);
    var element = document.createElement('a');
    element.setAttribute('href', "data:text/json;charset=UTF-8," + encodeURIComponent(sJson));
    element.setAttribute('download', this.currentExperiment.experiment.name+".json");
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  }

  loadPreviousExperiment(){
    this.openDialog()
  }

  onFileSelected(event:any) {
    const file:File = event.target.files[0];
    const fr = new FileReader();
    fr.onload = () => {
      if(fr.result != null){this.fileContent = fr.result};
      console.log('Commands', this.fileContent);
      this.setExperimentWithJSON();
    }
      if (file) {
          this.fileName = file.name;
          const formData = new FormData();
          formData.append("thumbnail", file);
          fr.readAsText(file);  
      }
  }

  setExperimentWithJSON(){
    let loadedExperiment = JSON.parse(this.fileContent.toString());
    this.currentExperiment.callbacks = loadedExperiment.callbacks;
    this.currentExperiment.experiment = loadedExperiment.experiment;
    this.currentExperiment.explorer = loadedExperiment.explorer;
    this.currentExperiment.input_wrappers = loadedExperiment.input_wrappers;
    this.currentExperiment.logger_handlers = loadedExperiment.logger_handlers;
    this.currentExperiment.output_representations = loadedExperiment.output_representations;
    this.currentExperiment.system = loadedExperiment.system;
    this.createNewExperimentService.setAllCustomModulesFromUseModule()
  }

  setExperimentWithPreviousExperiment(id : number){
    this.appDBService.getExperimentById(id).subscribe((experiment) => {
      this.currentExperiment.callbacks = [];
      this.currentExperiment.experiment.name = experiment.name;
      this.currentExperiment.experiment.config = experiment.config;
      this.currentExperiment.explorer = experiment.explorers[0];
      this.currentExperiment.input_wrappers = experiment.input_wrappers;
      this.currentExperiment.logger_handlers = [];
      this.currentExperiment.output_representations = experiment.output_representations;
      this.currentExperiment.system = experiment.systems[0];
      this.createNewExperimentService.setAllCustomModulesFromUseModule()
    });
  }

}
