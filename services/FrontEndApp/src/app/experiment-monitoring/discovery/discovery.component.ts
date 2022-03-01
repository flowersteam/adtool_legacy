import { Component, OnInit, Input} from '@angular/core';

import { ExpeDbService } from '../../services/expe-db.service';
import{ NumberUtilsService } from '../../services/number-utils.service'

@Component({
  selector: 'app-discovery',
  templateUrl: './discovery.component.html',
  styleUrls: ['./discovery.component.scss']
})
export class DiscoveryComponent implements OnInit {

  @Input() experiment?: any;

  currentRunIdx:any = [];           // run_idx value defined by the user. Using to show discovery of this specific run_idx
  allSeedCheckoxSelect:any = [];    // seed value defined by the user to visualise it in discoveries tab
  nbDiscoveriesDisplay = 12;        // how many discories we want display simultaneously
  indexDiscoveriesDisplay:number=0; //index to define wich subarray of run_idx we want display now
  arrayFilterRunIdx:any = [];       //subarray of run_idx we want see now
  sliderDoubleValue = {
    value: 0,
    highValue: 0,
    options: {
      floor: 0,
      ceil: 0
   }
  }

  constructor(private expeDbService: ExpeDbService, public numberUtilsService : NumberUtilsService) { }

  ngOnInit(): void {
  }

  ngOnChanges() : void{
    this.refreshDiscoveries();
  }

  setCurrentRunIdx(){
    this.currentRunIdx = [];
    for (let index = this.sliderDoubleValue.value; index <= this.sliderDoubleValue.highValue; index++) {
        this.currentRunIdx.push(index)
    }
  }

  defineWhatWeWantVisualise(){
    this.indexDiscoveriesDisplay = 0
    this.setCurrentRunIdx();
    this.getDiscovery();
  }

  definedFilters(): string{
    let filter = "";
    if(this.experiment){
      this.arrayFilterRunIdx = [];
      for (let i = 0; i <= Math.floor(this.currentRunIdx.length / this.nbDiscoveriesDisplay); i++) {
        this.arrayFilterRunIdx.push(this.currentRunIdx.slice(i*this.nbDiscoveriesDisplay, (i+1)*this.nbDiscoveriesDisplay))
        if(this.arrayFilterRunIdx[i].length == 0){
          this.arrayFilterRunIdx.splice(i, 1); 
        }
      }
      filter = '{"$and":[{"experiment_id":'
                    +this.experiment.id.toString()
                    +'}, {"run_idx":{"$in":'
                    +JSON.stringify(this.arrayFilterRunIdx[this.indexDiscoveriesDisplay])
                    +'}},  {"seed":{"$in":'
                    +JSON.stringify(this.allSeedCheckoxSelect)
                    +'}}]}'
    }
    return filter;
  }
  
  setIndexDiscoveries(i:number){
    this.indexDiscoveriesDisplay = this.indexDiscoveriesDisplay + i;
    if(i == 0){
      this.indexDiscoveriesDisplay = 0;
    }
    else if(this.indexDiscoveriesDisplay < 0){
      this.indexDiscoveriesDisplay =this.arrayFilterRunIdx.length -1;
    }
    else if(this.indexDiscoveriesDisplay >= this.arrayFilterRunIdx.length){
      this.indexDiscoveriesDisplay = 0;
    }
    this.getDiscovery();
  }

  getDiscovery(): void {
    if(this.experiment){
      for(let index = 0; index < this.experiment.config.nb_seeds; index++){
          let video = <HTMLVideoElement><any> document.querySelector("#video_"+index.toString());
          if(video){
            video.src = "";
          }
      }
      let filter = this.definedFilters()
      this.expeDbService.getDiscovery(filter)
      .subscribe(discoveries => {
        if(discoveries.length > 0){
          for(let discoverie of discoveries){
            this.expeDbService.getDiscoveryRenderedOutput(discoverie._id)
            .subscribe((renderedOutput : any) => {
              let video = <HTMLVideoElement> <any> document.querySelector("#video_"+discoverie.seed.toString()+"_"+discoverie.run_idx.toString());
              if(video){
                video.src = window.URL.createObjectURL(renderedOutput);
              } 
            });
          }
          
        }
        
      });
    }
  }

  refreshDiscoveries(){
    this.sliderDoubleValue = {
      value: this.sliderDoubleValue.value,
      highValue: this.sliderDoubleValue.highValue,
      options: {
        floor: 0,
        ceil: this.experiment ? this.experiment.progress-1 : this.sliderDoubleValue.options.ceil 
      }
    }
    this.defineWhatWeWantVisualise();
  }
}
