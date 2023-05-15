import { Component, OnInit, Input } from '@angular/core';
import { CreateNewExperimentService } from '../../services/create-new-experiment.service';

@Component({
  selector: 'app-set-experiment-config',
  templateUrl: './set-experiment-config.component.html',
  styleUrls: ['./set-experiment-config.component.scss'],
})
export class SetExperimentConfigComponent implements OnInit {
  objectKeys = Object.keys;

  @Input() currentConfig?: any;
  @Input() hosts?: any;

  constructor(public createNewExperimentService: CreateNewExperimentService) {}

  ngOnInit(): void {}
}
