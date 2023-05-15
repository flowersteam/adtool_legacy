import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DiscoveryVisComponent } from './discovery-vis.component';

describe('DiscoveryVisComponent', () => {
  let component: DiscoveryVisComponent;
  let fixture: ComponentFixture<DiscoveryVisComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DiscoveryVisComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(DiscoveryVisComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
