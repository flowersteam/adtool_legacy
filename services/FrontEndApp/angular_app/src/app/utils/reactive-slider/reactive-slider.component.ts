import { Component, EventEmitter, OnInit } from '@angular/core';
import { Input, Output } from '@angular/core';
import { Observable, BehaviorSubject } from 'rxjs';
import { tap } from 'rxjs/operators';
import { Options } from '@angular-slider/ngx-slider';

@Component({
  selector: 'app-reactive-slider',
  templateUrl: './reactive-slider.component.html',
  styleUrls: ['./reactive-slider.component.scss'],
})
export class ReactiveSliderComponent implements OnInit {
  // Output events to change another component's state
  @Output() lowValueChange = new EventEmitter();
  @Output() highValueChange = new EventEmitter();

  // Input Observables which are watched
  @Input() reactiveMax?: Observable<number> = new BehaviorSubject(1);
  @Input() reactiveMin: Observable<number> = new BehaviorSubject(0);
  options: Options = {
    floor: 0,
    ceil: 1,
  };
  // by default, set the slider to the Options defaults
  lowValue: number = this.options.floor!;
  highValue: number = this.options.ceil!;
  // slider will track reactive Options changes unless user has changed
  userHasChangedSlider: boolean = false;

  ngOnInit(): void {
    // hook observable into options
    this.reactiveMax?.pipe(tap(console.log)).subscribe((max) => {
      this.reloadOptions({ ceil: max });
      if (!this.userHasChangedSlider) {
        // only propagate the options change if user has not touched
        this.highValue = max;
      }
    });
    this.reactiveMin?.subscribe((min) => {
      this.reloadOptions({ floor: min });
      if (!this.userHasChangedSlider) {
        // only propragate the options change if user has not touched
        this.highValue = min;
      }
    });
  }

  private reloadOptions({ floor, ceil }: Options): void {
    const newOptions: Options = Object.assign({}, this.options);
    if (ceil !== undefined) {
      newOptions.ceil = ceil;
    }
    if (floor !== undefined) {
      newOptions.floor = floor;
    }
    this.options = newOptions;
  }
}
