import { Component, OnInit, Input } from '@angular/core';

@Component({
  selector: 'app-display-inputspace-outputspace',
  templateUrl: './display-inputspace-outputspace.component.html',
  styleUrls: ['./display-inputspace-outputspace.component.scss'],
})
export class DisplayInputspaceOutputspaceComponent implements OnInit {
  objectKeys = Object.keys;

  @Input() inputList?: any;
  @Input() outputList?: any;
  @Input() inputOutput?: any;

  constructor() {}

  ngOnInit(): void {}
}
