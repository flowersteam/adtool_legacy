import { HttpClient } from '@angular/common/http';
import { Component, OnInit, Input } from '@angular/core';

import { CreateNewExperimentService } from '../../services/create-new-experiment.service';

@Component({
  selector: 'app-load-experiment-config-to-create',
  templateUrl: './load-experiment-config-to-create.component.html',
  styleUrls: ['./load-experiment-config-to-create.component.scss']
})
export class LoadExperimentConfigToCreateComponent implements OnInit {

  @Input() currentExperiment?: any;

  fileName = '';
  fileContent : string | ArrayBuffer = '';
  constructor(private http: HttpClient, public createNewExperimentService: CreateNewExperimentService) { }
  
  ngOnInit(): void {}

  downloadJson(){
    var sJson = JSON.stringify(this.currentExperiment);
    var element = document.createElement('a');
    element.setAttribute('href', "data:text/json;charset=UTF-8," + encodeURIComponent(sJson));
    element.setAttribute('download', this.currentExperiment.experiment.name);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
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

}
