import { Component, OnInit, Input } from '@angular/core';
import { ExperimentCreationComponent } from '../experiment-creation.component';
import { CreateNewExperimentService } from '../../services/create-new-experiment.service';

import { Observable, of, Subject } from 'rxjs';

@Component({
  selector: 'app-set-module',
  templateUrl: './set-module.component.html',
  styleUrls: ['./set-module.component.scss'],
})
export class SetModuleComponent implements OnInit {
  objectKeys = Object.keys;

  @Input() currentModule?: any; // return by reference

  @Input() modules?: any;
  @Input() moduleItDependsOn?: any;
  @Input() displayInputOutputSpace?: Boolean;

  constructor(public createNewExperimentService: CreateNewExperimentService) {}

  ngOnInit(): void {}
}
