import { Component, OnInit, Input, HostListener } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import * as bootstrap from 'bootstrap';

import { JupyterService } from '../../../services/jupyter.service';
import { JupyterSessions } from '../../../entities/jupyter_sessions';

@Component({
  selector: 'app-jupyter-lab',
  templateUrl: './jupyter-lab.component.html',
  styleUrls: ['./jupyter-lab.component.scss'],
})
export class JupyterLabComponent implements OnInit {
  @Input() experiment?: any;

  public jupyterSession: JupyterSessions[] = [];
  public currentSessionPath: string | undefined;
  public kernelInfoSet = false;
  public message: string = '';
  aKernelToRuleThemAll: any;
  needSendMessageToKernel = true;

  urlSafe: SafeResourceUrl | undefined;

  constructor(
    public sanitizer: DomSanitizer,
    private jupyterService: JupyterService
  ) {}

  ngOnInit(): void {
    this.jupyterService.createKernel().subscribe((res) => {
      this.aKernelToRuleThemAll = res;
      this.jupyterService
        .openKernelChannel(this.aKernelToRuleThemAll.id)
        .subscribe((_) => {
          if (this.needSendMessageToKernel) {
            this.makeKernelMessageToCreateDataset();
            this.jupyterService.sendToKernel(this.message);
            this.needSendMessageToKernel = false;
          }
        });
    });
    this.defineJupyterLab();
    this.initPopover();
  }

  ngOnChanges(): void {
    this.defineJupyterLab();
  }

  initPopover() {
    var popoverTriggerList = [].slice.call(
      document.querySelectorAll('[data-bs-toggle="popover"]')
    );
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
      return new bootstrap.Popover(popoverTriggerEl);
    });
  }

  defineInfoToAccessJupyterLab(exp_name: string, exp_id: number) {
    let path = exp_name + '_' + exp_id.toString();
    this.urlSafe = this.sanitizer.bypassSecurityTrustResourceUrl(
      this.jupyterService.jupyterUrl +
        '/lab/workspaces/' +
        path +
        '/tree/Experiments/' +
        path +
        '/'
    );
    this.currentSessionPath = exp_name + '_' + exp_id.toString() + '.ipynb';
    this.kernelInfoSet = true;
  }

  makeKernelMessageToCreateDataset() {
    if (this.experiment) {
      console.log('Creating dataset');
      this.message =
        'from auto_disc_db import Dataset' +
        '\n' +
        'if __name__ == "__main__":' +
        '\n' +
        '     dataset_' +
        this.experiment.id.toString() +
        ' = Dataset(' +
        this.experiment.id.toString() +
        ')' +
        '\n' +
        '     dataset = Dataset(' +
        this.experiment.id.toString() +
        ')' +
        '\n' +
        '     print(dataset)' +
        '\n' +
        '     %store dataset_' +
        this.experiment.id.toString() +
        '\n' +
        '     %store dataset';
    }
  }

  defineJupyterLab() {
    if (!this.kernelInfoSet && this.experiment) {
      this.defineInfoToAccessJupyterLab(
        this.experiment.name,
        this.experiment.id
      );
    }
  }

  @HostListener('window:beforeunload')
  ngOnDestroy(): void {
    this.jupyterService.destroyKernel(this.aKernelToRuleThemAll.id).subscribe();
    this.jupyterService.closeKernelChannel();
  }
}
