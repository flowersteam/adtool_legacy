import { Component, OnInit, Input, Output,EventEmitter  } from '@angular/core';

@Component({
  selector: 'app-filter',
  templateUrl: './filter.component.html',
  styleUrls: ['./filter.component.scss']
})
export class FilterComponent implements OnInit {

  constructor() { }
  objectKeys = Object.keys;
  shareFilters = true;
  @Input() filters?: any;
  @Input() useFilters?: any;
  @Output() useKeyListChange = new EventEmitter();
  @Output() triggerParentMethod = new EventEmitter<any>();
  
  ngOnInit(): void {
  }

  fromListToOther(originList:any, arrivalList:any, index?:number, removeAll?:boolean){
    if(index != undefined){
      arrivalList.push(originList[index]);
      originList.splice(index, 1);
    }
    else{
      while(originList.length > 0){
        arrivalList.push(originList[0]);
        originList.splice(0, 1);
      }
    }
    arrivalList = this.sortFilters(arrivalList);
    originList = this.sortFilters(originList);
    if(removeAll == undefined || !removeAll){
      this.triggerParentMethod.next();
    }
  }

  removeAllFilters(){
    for(let key of  Object.keys(this.filters)){
      this.fromListToOther(this.useFilters[key], this.filters[key],undefined, true)
    }
    this.triggerParentMethod.next();
  }

  test(){
    console.log("oui");
  }

  isFilterUse(){
    let size : number = 0;
    for(let key of Object.keys(this.useFilters)){
      size = size + this.useFilters[key].length;
    }
    return size == 0;
  }

  sortFilters(filtersList: any){
    if(typeof(filtersList[0]) == 'string'){
      filtersList = filtersList.sort((n1: any,n2: any) => {
        if (n1 > n2) {
            return 1;
        }
    
        if (n1 < n2) {
            return -1;
        }
    
        return 0;
      });
    }else if(typeof(filtersList[0]) == 'number'){
      filtersList = filtersList.sort((n1: number,n2: number) => n1 - n2);
    }
    return filtersList;
  }

}
