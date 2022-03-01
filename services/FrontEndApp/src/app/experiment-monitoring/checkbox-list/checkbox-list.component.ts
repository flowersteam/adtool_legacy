import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import{ NumberUtilsService } from '../../services/number-utils.service'

@Component({
  selector: 'app-checkbox-list',
  templateUrl: './checkbox-list.component.html',
  styleUrls: ['./checkbox-list.component.scss']
})
export class CheckboxListComponent implements OnInit {

  @Input() keyList?: any;
  @Input() useKeyList?: any;
  @Input() checkboxName?: string;
  @Output() useKeyListChange = new EventEmitter();

  isAllSelect : Boolean = false;
  constructor(public numberUtilsService : NumberUtilsService) { }

  ngOnInit(): void {}

  ngOnChanges() : void{
    this.isAllSelect = this.keyList.length == this.useKeyList.length;
  }

  setCurrentCheckbox(){   
    this.useKeyList = []
    let checkbox = document.querySelectorAll('input[name="'+this.checkboxName+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      if((checkbox[index]as HTMLInputElement).checked){
        let res = checkbox[index].id.replace(this.checkboxName+'','');
        this.useKeyList.push(this.keyList[parseInt(res)])      
      } 
    }
    this.isAllSelect = this.keyList.length == this.useKeyList.length;
    this.useKeyListChange.emit(this.useKeyList);
  }

  selectAllCheckbox(){
    let checkbox = document.querySelectorAll('input[name="'+this.checkboxName+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      (checkbox[index]as HTMLInputElement).checked = true;
    }
    this.setCurrentCheckbox();
  }

  unselectAllCheckbox(){
    let checkbox = document.querySelectorAll('input[name="'+this.checkboxName+'"]')
    for (let index = 0; index < checkbox.length; index++) {
      (checkbox[index]as HTMLInputElement).checked = false;
    }
    this.setCurrentCheckbox();
  }

  manegeAllCheckbox(){
    if(this.isAllSelect){
      this.unselectAllCheckbox();
    }
    else{
      this.selectAllCheckbox();
    }
  }
}
