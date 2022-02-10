import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'app-select-input-according-to-type',
  templateUrl: './select-input-according-to-type.component.html',
  styleUrls: ['./select-input-according-to-type.component.scss']
})
export class SelectInputAccordingToTypeComponent implements OnInit {

  @Input() elt_config_key?: string;
  @Input() actual_config_elt?: any;
  @Input() inputValue?: any;

  @Output() inputValueChange = new EventEmitter();

  constructor() { }

  ngOnInit(): void {}
  
  calculStep(){
    return(Math.trunc((this.actual_config_elt.max - this.actual_config_elt.min)/20))
  }

  parseToDict(){
    this.inputValue = JSON.parse(this.inputValue);
  }

  return_to_parent(){
    this.inputValueChange.emit(this.inputValue);
  }

}
