import { Component, OnInit, Input } from '@angular/core';

import { AppDbService } from '../../../services/REST-services/app-db.service';
import { NumberUtilsService } from '../../../services/number-utils.service';
import { ToasterService } from '../../../services/toaster.service';

@Component({
  selector: 'app-logs',
  templateUrl: './logs.component.html',
  styleUrls: ['./logs.component.scss'],
})
export class LogsComponent implements OnInit {
  constructor(
    private appDBService: AppDbService,
    private toasterService: ToasterService,
    public numberUtilsService: NumberUtilsService
  ) {}

  @Input() experiment?: any;

  logsLevel: any = [];

  useFilters: { [key: string]: any[] } = {
    checkpoints: <any>[],
    seeds: <any>[],
    levels: <any>[],
  };

  allFilters: { [key: string]: any[] } = {
    checkpoints: <any>[],
    seeds: <any>[],
    levels: <any>[],
  };

  logsValue: any = [];

  ngOnInit(): void {
    this.getLogLevels();
  }

  ngOnChanges() {
    this.logsWewant();
    if (this.experiment) {
      this.allFilters['seeds'] = this.numberUtilsService.nFirstIntegers(
        this.experiment.config.nb_seeds
      );
      this.allFilters['checkpoints'] = this.getAttributAsList(
        this.experiment.checkpoints,
        'id'
      );
    }
  }

  getLogLevels() {
    this.appDBService.getAllLogLevels().subscribe((response) => {
      if (response.success) {
        this.logsLevel = response.data;
        this.allFilters['levels'] = this.getAttributAsList(
          this.logsLevel,
          'name'
        );
      } else {
        this.toasterService.showError(
          response.message ?? '',
          'Error getting log levels'
        );
      }
    });
  }

  definedOneFilterParam(param: string, param_name: string) {
    param = param.replace('[', '(');
    param = param.replace(']', ')');
    if (param.length <= 2) {
      param = '';
    } else {
      param = '&' + param_name + '=in.' + param;
    }
    return param;
  }

  logsWewant() {
    if (this.experiment) {
      let checkpoints = this.definedOneFilterParam(
        JSON.stringify(this.useFilters['checkpoints']),
        'checkpoint_id'
      );
      let seeds = this.definedOneFilterParam(
        JSON.stringify(this.useFilters['seeds']),
        'seed'
      );
      let log_levels = this.definedOneFilterParam(
        JSON.stringify(
          this.fromLogsLevelsNameToLogsLevelsIds(this.useFilters['levels'])
        ),
        'log_level_id'
      );
      let filter =
        '?&experiment_id=eq.' +
        this.experiment.id.toString() +
        checkpoints +
        log_levels +
        seeds;
      this.appDBService.getLogs(filter).subscribe((response) => {
        if (response.success) {
          this.logsValue = response.data;
        } else {
          this.toasterService.showError(
            response.message ?? '',
            'Error getting logs'
          );
        }
      });
    }
  }

  getAttributAsList(currentList: any, attribut: string) {
    let logsLevelsNames = [];
    for (let elt of currentList) {
      logsLevelsNames.push(elt[attribut]);
    }
    return logsLevelsNames;
  }

  fromLogsLevelsNameToLogsLevelsIds(logsLevelsNames: any) {
    let logsLevelsIds = [];
    for (let logLevelNames of logsLevelsNames) {
      for (let log of this.logsLevel) {
        if (log.name == logLevelNames) {
          logsLevelsIds.push(log.id);
        }
      }
    }
    return logsLevelsIds;
  }
}
