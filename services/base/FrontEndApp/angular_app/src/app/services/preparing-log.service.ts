import { Injectable } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { Router } from '@angular/router';
import { PreparingLogComponent } from '../utils/preparing-log/preparing-log.component';

@Injectable({
  providedIn: 'root',
})
export class PreparingLogService {
  constructor(public dialog: MatDialog, private router: Router) {}
  dialogRef: any;

  openDialog(id: number): void {
    this.dialogRef = this.dialog.open(PreparingLogComponent, {
      data: { experiment_id: id },
      disableClose: true,
    });
    this.router.events.subscribe(() => {
      this.dialogRef.close();
    });
    this.dialogRef.afterClosed().subscribe((result: undefined) => {
      if (result != undefined) {
      }
    });
  }
  closeDialog() {
    this.dialogRef.close();
  }

  sleep(ms: number) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }
}
