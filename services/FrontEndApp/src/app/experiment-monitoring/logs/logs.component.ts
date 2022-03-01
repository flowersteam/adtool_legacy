import { Component, OnInit, Input } from '@angular/core';
import  * as bootstrap  from 'bootstrap'

import { AppDbService } from '../../services/app-db.service';
import{ NumberUtilsService } from '../../services/number-utils.service'

@Component({
  selector: 'app-logs',
  templateUrl: './logs.component.html',
  styleUrls: ['./logs.component.scss']
})
export class LogsComponent implements OnInit {

  constructor(private appDBService: AppDbService, public numberUtilsService : NumberUtilsService) { }

  @Input() experiment?: any;

  logsLevel :any = [];

  checkBoxList : { [key: string]: any []; }={
    checkpoints :<any>[],
    seeds :<any>[],
    levels :<any>[],
  }
  
  logsValue :any = [];

  ngOnInit(): void {
    this.getLogLevels();
  }

  getLogLevels(){
    this.appDBService.getAllLogLevels().subscribe(res =>{ this.logsLevel = res});
  }

  definedOneFilterParam(param:string, param_name:string){
    param = param.replace("[", "(");
    param = param.replace("]", ")");
    if(param.length <= 2){
      param = ""
    }
    else{
      param = "&"+param_name+"=in."+param
    }
    return(param)
  }

  logsWewant(){
    if(this.experiment){
      let checkpoints = this.definedOneFilterParam(JSON.stringify(this.checkBoxList["checkpoints"]), 'checkpoint_id');
      let seeds = this.definedOneFilterParam(JSON.stringify(this.checkBoxList["seeds"]), 'seed');
      let log_levels = this.definedOneFilterParam(JSON.stringify(this.fromLogsLevelsNameToLogsLevelsIds(this.checkBoxList["levels"])), 'log_level_id');
      let filter = "?&experiment_id=eq."+this.experiment.id.toString() + checkpoints + log_levels + seeds;
      this.appDBService.getLogs(filter).subscribe(res => this.logsValue = res);
    }
  }

  collapseLogs(event: any){
    var collapseTriggerList:any = [].slice.call(document.querySelectorAll('#collapseCheckBoxSeedLogs, #collapseCheckBoxCheckpointLogs, #collapseCheckBoxLevelLogs'))
    var collapseBtnTriggeringList =[];
    for(let collapseTrigger of collapseTriggerList){
      if(event.delegateTarget.id.replace("btn", "") == collapseTrigger.id || collapseTrigger.classList.value.includes('show')){
        collapseBtnTriggeringList.push(collapseTrigger);
      }
    }
     
    var collapseList = collapseBtnTriggeringList.map(function (collapseTriggerEl) {
      return new bootstrap.Collapse(collapseTriggerEl)
    })
  }

  getAttributAsList(currentList : any, attribut : string){
    let logsLevelsNames = [];
    for(let elt of currentList){
      logsLevelsNames.push(elt[attribut]);
    }
    return logsLevelsNames;
  }

  fromLogsLevelsNameToLogsLevelsIds(logsLevelsNames : any){
    let logsLevelsIds = [];
    for(let logLevelNames of logsLevelsNames){
      for(let log of this.logsLevel){
        if(log.name == logLevelNames){
          logsLevelsIds.push(log.id);
        }
      }
    }
    return logsLevelsIds;
  }
}