import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ReactiveSliderComponent } from './reactive-slider.component';

describe('ReactiveSliderComponent', () => {
  let component: ReactiveSliderComponent;
  let fixture: ComponentFixture<ReactiveSliderComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ReactiveSliderComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(ReactiveSliderComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
