import { Component, Input, OnInit} from '@angular/core';
import  * as bootstrap  from 'bootstrap'

@Component({
  selector: 'app-multi-collapser',
  templateUrl: './multi-collapser.component.html',
  styleUrls: ['./multi-collapser.component.scss']
})
export class MultiCollapserComponent implements OnInit {

  @Input() experiment?: any;

  childComponentName = ["Jupyter", "Discovery", "Logs"];
  tabButtonDisable:any = {};
  constructor() { }

  ngOnInit(): void {
    this.initCollapseVisualisation();
  }

  initCollapseVisualisation(){
    for(let name of this.childComponentName){
      this.tabButtonDisable["btncollapse"+name] = false;
    }
    this.tabButtonDisable["btncollapse"+this.childComponentName[0]] = true;
  }

  fromArrayNameToQueryString(){
    let querySelector = "";
    for(let name of this.childComponentName){
      querySelector = querySelector +"#collapse" + name + ", ";
    }
    return(querySelector.slice(0, -2));
  }

  collapseVisualisation(event: any){
    var collapseTriggerList:any = [].slice.call(document.querySelectorAll(this.fromArrayNameToQueryString()));
    var collapseBtnTriggeringList =[];
    for(let collapseTrigger of collapseTriggerList){
      if(event.delegateTarget.id.replace("btn", "") == collapseTrigger.id || collapseTrigger.classList.value.includes('show')){
        collapseBtnTriggeringList.push(collapseTrigger);
      }
    }
     
    var collapseList = collapseBtnTriggeringList.map(function (collapseTriggerEl) {
      return new bootstrap.Collapse(collapseTriggerEl)
    })

    for (let key in this.tabButtonDisable) {
      this.tabButtonDisable[key] = false;
    }
    this.tabButtonDisable[event.delegateTarget.id] = true;
  }
}