import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

import { AutoDiscServerService } from '../../services/REST-services/auto-disc.service';
import { ToasterService } from '../../services/toaster.service';
import { CreateNewExperimentService } from 'src/app/services/create-new-experiment.service';
import { ExperimentSettings } from 'src/app/entities/experiment_settings';
@Component({
  selector: 'app-set-discovery-saving-key',
  templateUrl: './set-discovery-saving-key.component.html',
  styleUrls: ['./set-discovery-saving-key.component.scss']
})
export class SetDiscoverySavingKeyComponent implements OnInit {

  objectKeys = Object.keys;

  @Input() saveCheckBox?: any;
  @Output() saveCheckBoxEventEmitter = new EventEmitter();

  @Input() system_name?: any;
  @Input() system_settings?: any;
  @Input() input_wrappers?: any;
  @Input() input_wrappers_settings?: any;
  @Input() output_representations?: any;
  @Input() output_representations_settings?: any;

  @Input() explorer_name: string | undefined = '';

  @Input() actual_config_elt?: any;

  discovery_saveflags: Map<string, boolean> = new Map();

  constructor(private AutoDiscServerService: AutoDiscServerService,
    private toasterService: ToasterService,
    private CreateNewExperimentService: CreateNewExperimentService) { }

  ngOnInit(): void {
    this.CreateNewExperimentService.stagedExperiment
      .subscribe((newSetting: ExperimentSettings) => {
        this.explorer_name = newSetting.explorer.name;
        this.updateDiscoverySpec();
      });
  }

  updateDiscoverySpec(): void {
    // reinitialize discovery_saveflags
    this.initDiscoveryToSave();
  }

  initDiscoveryToSave(): void {
    // null guard
    if (this.explorer_name) {
      let discovery_spec: string[] = [];
      this.AutoDiscServerService
        .getDiscoverySavingKeys(this.explorer_name as string)
        .subscribe(response => {

          if (response.success) {
            // reinitialize labels for checkbox
            discovery_spec = response.data ?? [];
            this.discovery_saveflags = new Map(discovery_spec.map(x => [x, true]));
            this.updateParentComponent();
          }

          else {
            this.toasterService.showError(response.message ?? '',
              "Error querying discovery specification",
              { timeOut: 0, extendedTimeOut: 0 });
          }

        });
    }
  }

  onCheckboxChange(key: string) {
    let flag = this.discovery_saveflags.get(key);
    this.discovery_saveflags.set(key, !flag);
  }

  updateParentComponent() {
    this.saveCheckBoxEventEmitter.emit(this.saveCheckBox);
  }

}
