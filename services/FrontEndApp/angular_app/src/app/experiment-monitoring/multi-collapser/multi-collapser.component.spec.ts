import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MultiCollapserComponent } from './multi-collapser.component';

describe('MultiCollapserComponent', () => {
  let component: MultiCollapserComponent;
  let fixture: ComponentFixture<MultiCollapserComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [MultiCollapserComponent],
    }).compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(MultiCollapserComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
