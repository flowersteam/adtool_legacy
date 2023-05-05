import { Component, OnInit } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef } from '@angular/material/dialog';
import { Inject } from '@angular/core';
import { AppDbService } from '../../services/REST-services/app-db.service';
import { Event, RouterEvent, Router } from '@angular/router';

@Component({
  selector: 'app-preparing-log',
  templateUrl: './preparing-log.component.html',
  styleUrls: ['./preparing-log.component.scss'],
})
export class PreparingLogComponent implements OnInit {
  constructor(
    private appDBService: AppDbService,
    public router: Router,
    @Inject(MAT_DIALOG_DATA) public data: any,
    private dialogRef: MatDialogRef<PreparingLogComponent>
  ) {}

  preparing_logs: any;
  interval: any;

  ngOnInit(): void {
    let filter = '?&experiment_id=eq.' + this.data.experiment_id.toString();
    let timeIntervalSeconds = 2;
    this.interval = setInterval(() => {
      this.appDBService
        .getPreparingLogs(filter)
        .subscribe((res: any) => (this.preparing_logs = res));
    }, timeIntervalSeconds * 1000);
    this.router.events.subscribe(() => {
      clearInterval(this.interval);
    });
  }

  ngOnDestroy() {
    clearInterval(this.interval);
  }

  closeDialog() {
    this.dialogRef.close();
  }
}
