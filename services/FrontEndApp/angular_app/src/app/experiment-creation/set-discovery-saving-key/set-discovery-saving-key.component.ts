import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

import { AutoDiscServerService } from '../../services/REST-services/auto-disc.service';
import { ToasterService } from '../../services/toaster.service';
@Component({
  selector: 'app-set-discovery-saving-key',
  templateUrl: './set-discovery-saving-key.component.html',
  styleUrls: ['./set-discovery-saving-key.component.scss']
})
export class SetDiscoverySavingKeyComponent implements OnInit {

  objectKeys = Object.keys;
  
  @Input() inputValueCheckBox?: any;
  @Output() inputValueCheckBoxChange = new EventEmitter();

  @Input() system_name?: any;
  @Input() system_settings?: any;
  @Input() input_wrappers?: any;
  @Input() input_wrappers_settings?: any;
  @Input() output_representations?: any;
  @Input() output_representations_settings?: any;


  @Input() actual_config_elt?: any;

  
  discovery_saving_keys: string[] = []
  discovery_saving_keys_use: { [name: string]: boolean } = {};

  constructor(private AutoDiscServerService: AutoDiscServerService, private toasterService: ToasterService) { }

  ngOnInit(): void {
    this.getDiscoverySavingKeys();
  }

  ngOnChanges(): void{
    for (let key of this.discovery_saving_keys) {
      this.discovery_saving_keys_use[key.toString()] = false;   
    }
    for(let elt of this.inputValueCheckBox){
      this.discovery_saving_keys_use[elt] = true;
    }
  }

  getDiscoverySavingKeys(): void {
    this.AutoDiscServerService.getDiscoverySavingKeys()
    .subscribe(response => {
      if(response.success) {
        let discovery_saving_keys = response.data ?? [];
        this.discovery_saving_keys = discovery_saving_keys.map(discovery_saving_key => discovery_saving_key.toString())
        this.initDiscoverySavingKeysUse(),
        this.getDiscoverySavingKeysUse()
      }
      else{
        this.toasterService.showError(response.message ?? '', "Error getting discovery saving keys", {timeOut: 0, extendedTimeOut: 0});
      }
    });
  }

  initDiscoverySavingKeysUse():void{
    for (let key of this.discovery_saving_keys) {
      this.discovery_saving_keys_use[key.toString()] = true;   
    }
  }

  getDiscoverySavingKeysUse(){
    this.inputValueCheckBox = []
    for (let key in this.discovery_saving_keys_use) {
      if (this.discovery_saving_keys_use[key]){
        this.inputValueCheckBox.push(key)
      }
    }
    this.returnToParent()
  }

  onCheckboxChange(key: string) {
    this.discovery_saving_keys_use[key] = !this.discovery_saving_keys_use[key]
    this.getDiscoverySavingKeysUse()  
  }

  

  returnToParent(){
    this.inputValueCheckBoxChange.emit(this.inputValueCheckBox);
  }

}
