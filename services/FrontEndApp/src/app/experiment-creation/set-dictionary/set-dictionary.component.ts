import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import  * as bootstrap  from 'bootstrap';

@Component({
  selector: 'app-set-dictionary',
  templateUrl: './set-dictionary.component.html',
  styleUrls: ['./set-dictionary.component.scss']
})
export class SetDictionaryComponent implements OnInit {

  @Input() inputValue?: any;
  @Output() inputValueChange = new EventEmitter();
  @Output() triggerParentMethod = new EventEmitter<any>();

  parseError = false;
  constructor() { }

  ngOnInit(): void {
    this.initPopover();
  }

  initPopover(){
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
      return new bootstrap.Popover(popoverTriggerEl)
    })
  }

  parseToDict(){
    this.inputValue = JSON.parse(this.inputValue);
  }

  parseAndReturnToParent(){
    try {
      let temporaryValue = JSON.parse(this.inputValue);
      this.inputValue = temporaryValue;
      this.parseError = false;
      this.returnToParent();
    } catch (error) {
      this.parseError = true;
    }
  }

  returnToParent(){
    this.inputValueChange.emit(this.inputValue);
    this.triggerParentMethod.next();
  }

}
