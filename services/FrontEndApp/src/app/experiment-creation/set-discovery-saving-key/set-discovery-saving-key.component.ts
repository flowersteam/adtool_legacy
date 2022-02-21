import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

import { AutoDiscServerService } from '../../services/auto-disc.service';
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

  constructor(private AutoDiscServerService: AutoDiscServerService) { }

  ngOnInit(): void {
    this.getDiscoverySavingKeys();
  }

  getDiscoverySavingKeys(): void {
    this.AutoDiscServerService.getDiscoverySavingKeys()
    .subscribe(discovery_saving_keys => {this.discovery_saving_keys = discovery_saving_keys.map(discovery_saving_key => discovery_saving_key.toString())
      this.setDiscoverySavingKeysUse(),
      this.getDiscoverySavingKeysUse()});
  }

  setDiscoverySavingKeysUse():void{
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
    this.return_to_parent()
  }

  onCheckboxChange(key: string) {
    this.discovery_saving_keys_use[key] = !this.discovery_saving_keys_use[key]
    this.getDiscoverySavingKeysUse()  
  }

  

  return_to_parent(){
    this.inputValueCheckBoxChange.emit(this.inputValueCheckBox);
  }

}
