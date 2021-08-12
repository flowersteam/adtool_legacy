import { Component, Input, OnInit, Output } from '@angular/core';
import { FormGroup, FormControl } from '@angular/forms';

import { AppDbService } from '../../services/app-db.service';
import { ExpeDbService } from '../../services/expe-db.service';
import { Experiment } from '../../entities/experiment';
import { Router } from '@angular/router';

@Component({
  selector: 'app-archive-experiment',
  templateUrl: './archive-experiment.component.html',
  styleUrls: ['./archive-experiment.component.scss']
})
export class ArchiveExperimentComponent implements OnInit {

  @Input() experiment?: Experiment;
  @Input() refreshExperimentMethod?: Function;

  archiveForm = new FormGroup({
    archiveExperiment: new FormControl(),
    archiveChekpointSaves: new FormControl(),
    archiveDiscoveries: new FormControl(),
  })

  constructor(private appDBService: AppDbService, private expeDBService: ExpeDbService, private router: Router) { }

  ngOnInit(): void {}

  updateForm(): void {
    if (this.experiment){
      this.archiveForm.controls['archiveExperiment'].valueChanges.subscribe((value) => {
        if (value){
          this.archiveForm.controls['archiveChekpointSaves'].setValue(true);
          this.archiveForm.controls['archiveChekpointSaves'].disable({emitEvent: false});
          this.archiveForm.controls['archiveDiscoveries'].setValue(true);
          this.archiveForm.controls['archiveDiscoveries'].disable({emitEvent: false});
        }
        else{
          this.archiveForm.controls['archiveChekpointSaves'].enable({emitEvent: false});
          this.archiveForm.controls['archiveChekpointSaves'].setValue(this.experiment?.checkpoint_saves_archived);
          this.archiveForm.controls['archiveDiscoveries'].enable({emitEvent: false});
          this.archiveForm.controls['archiveDiscoveries'].setValue(this.experiment?.discoveries_archived);
        }
      });

      this.archiveForm.controls['archiveChekpointSaves'].valueChanges.subscribe((value) => {
        if (value && value == this.experiment?.checkpoint_saves_archived){
          this.archiveForm.controls['archiveChekpointSaves'].disable({emitEvent: false});
        }
      });

      this.archiveForm.controls['archiveDiscoveries'].valueChanges.subscribe((value) => {
        if (value && value == this.experiment?.discoveries_archived){
          this.archiveForm.controls['archiveDiscoveries'].disable({emitEvent: false});
        }
      });
      
      this.archiveForm.controls['archiveChekpointSaves'].setValue(this.experiment.checkpoint_saves_archived);
      this.archiveForm.controls['archiveExperiment'].setValue(this.experiment.archived);
      this.archiveForm.controls['archiveDiscoveries'].setValue(this.experiment.discoveries_archived);
    }
    
  }

  applyRemoval(): void {
    if (this.experiment){
      var archiveCheckpointSavesValue = this.archiveForm.controls['archiveChekpointSaves'].value as boolean
      if (archiveCheckpointSavesValue && archiveCheckpointSavesValue != this.experiment.checkpoint_saves_archived){
        this.experiment.checkpoints.forEach(checkpoint => {
          this.expeDBService.deleteCheckpointSaves(checkpoint.id)
          .subscribe((result) => {console.log("Removed checkpoint save for checkpoint n°" + checkpoint.id);});
        });
        this.appDBService.archiveExperimentCheckpointSavesById(this.experiment.id)
        .subscribe((result) => {if(this.refreshExperimentMethod) this.refreshExperimentMethod()});
      }

      var archiveDiscoveriesValue = this.archiveForm.controls['archiveDiscoveries'].value as boolean
      if (archiveDiscoveriesValue && archiveDiscoveriesValue != this.experiment.discoveries_archived){
        this.experiment.checkpoints.forEach(checkpoint => {
          this.expeDBService.deleteCheckpointDiscoveries(checkpoint.id)
          .subscribe((result) => {console.log("Removed discoveries for checkpoint n°" + checkpoint.id);});
        });
        this.appDBService.archiveExperimentDiscoveriesById(this.experiment.id)
        .subscribe((result) => {if(this.refreshExperimentMethod) this.refreshExperimentMethod()});
      }

      var archiveExperimentValue = this.archiveForm.controls['archiveExperiment'].value as boolean
      if (archiveExperimentValue != this.experiment.archived){
        this.appDBService.updateArchiveExperimentStatusById(this.experiment.id, archiveExperimentValue)
        .subscribe((result) => {
          if(this.refreshExperimentMethod) this.refreshExperimentMethod();
          if (archiveExperimentValue){
            this.router.navigate(["/home"]);
          }
        });
      }
    }
  }
}
