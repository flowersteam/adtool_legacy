import { Component, Input, OnInit } from '@angular/core';
import { Experiment } from 'src/app/entities/experiment';
import { BehaviorSubject } from 'rxjs';

@Component({
  selector: 'app-discovery-vis',
  templateUrl: './discovery-vis.component.html',
  styleUrls: ['./discovery-vis.component.scss'],
})
export class DiscoveryVisComponent implements OnInit {
  @Input() experiment?: Experiment;
  @Input() media$!: BehaviorSubject<Map<[number, number], string>>;
  seeds?: number;
  run_idx?: number;
  media_src_mapping: String = '';

  ngOnInit(): void {
    this.seeds = this.experiment?.config?.nb_seeds;
    this.run_idx = this.experiment?.config?.nb_iterations;
  }
}
