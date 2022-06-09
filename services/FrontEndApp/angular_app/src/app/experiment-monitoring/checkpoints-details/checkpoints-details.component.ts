import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-checkpoints-details',
  templateUrl: './checkpoints-details.component.html',
  styleUrls: ['./checkpoints-details.component.scss']
})
export class CheckpointsDetailsComponent implements OnInit {

  @Input() experiment?: any;

  constructor() { }

  ngOnInit(): void {
  }

}
